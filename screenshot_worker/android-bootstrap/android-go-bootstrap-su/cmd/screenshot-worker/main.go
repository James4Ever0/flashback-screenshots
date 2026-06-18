package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/rs/zerolog/log"
	"github.com/spf13/cobra"

	"android-go-bootstrap-su/internal/briefing"
	"android-go-bootstrap-su/internal/buffer"
	"android-go-bootstrap-su/internal/config"
	"android-go-bootstrap-su/internal/logging"
	"android-go-bootstrap-su/internal/root"
	"android-go-bootstrap-su/internal/screencap"
	"android-go-bootstrap-su/internal/session"
	"android-go-bootstrap-su/internal/transport"
)

var (
	configPath string
	rootCmd    = &cobra.Command{
		Use:   "screenshot-worker",
		Short: "Screenshot worker for rooted Android (su)",
	}
)

func init() {
	rootCmd.PersistentFlags().StringVarP(&configPath, "config", "c", "", "config file path")

	rootCmd.AddCommand(
		daemonCmd,
		testRootCmd,
		testScreencapCmd,
		testLockCmd,
		statusCmd,
	)
}

// ── daemon ────────────────────────────────────────────────────────────────

var daemonCmd = &cobra.Command{
	Use:   "daemon",
	Short: "Run the screenshot worker daemon",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := config.Load(configPath)
		if err != nil {
			return fmt.Errorf("failed to load config: %w", err)
		}

		if err := logging.Setup(cfg.Client.LogLevel, cfg.Client.LogFile); err != nil {
			return fmt.Errorf("failed to setup logging: %w", err)
		}

		// Require root before anything else
		if err := root.RequireRoot(cfg.Root.AutoRequest, cfg.Root.RetryIntervalSec, cfg.Root.MaxRetrySec); err != nil {
			return fmt.Errorf("root check failed: %w", err)
		}
		log.Info().Msg("root access confirmed")

		log.Info().
			Str("name", cfg.Client.Name).
			Str("server", cfg.Server.URL).
			Int("display", cfg.Capture.DisplayID).
			Msg("starting daemon")

		// Disk buffer
		buf, err := buffer.New(cfg.Buffer.Path, cfg.Buffer.MaxFiles)
		if err != nil {
			return fmt.Errorf("failed to create buffer: %w", err)
		}
		log.Info().Str("dir", cfg.Buffer.Path).Int("max_files", cfg.Buffer.MaxFiles).Msg("buffer ready")

		// Transport
		client := transport.NewClient(cfg.Server.URL, cfg.Client.Name, buf, cfg.Upload.WifiOnly)

		// Briefing emitter
		emitter := briefing.NewEmitter(cfg.Briefing.IntervalSec, cfg.Client.Name, cfg.Capture.DisplayID, cfg.Upload.WifiOnly)

		// Context for graceful shutdown
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		// Start capture loop
		captureStop := make(chan struct{})
		go captureLoop(ctx, cfg, buf, emitter, captureStop)

		// Start transport
		go func() {
			if err := client.Run(ctx); err != nil {
				log.Error().Err(err).Msg("transport exited")
			}
		}()

		// Start briefing emitter
		go emitter.Run(ctx)

		// Sync transport stats into briefing
		go func() {
			ticker := time.NewTicker(time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					emitter.WSConnected = client.Connected()
					emitter.FramesSent = client.FramesSent()
					emitter.LastSendMs = client.LastSendTime()
				}
			}
		}()

		// Wait for interrupt
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh

		log.Info().Msg("shutting down")
		cancel()
		client.Stop()
		<-captureStop
		time.Sleep(100 * time.Millisecond)
		return nil
	},
}

func captureLoop(ctx context.Context, cfg config.Config, buf *buffer.Buffer, emitter *briefing.Emitter, done chan<- struct{}) {
	defer close(done)

	ticker := time.NewTicker(time.Duration(cfg.Capture.IntervalSec) * time.Second)
	defer ticker.Stop()

	var captureCount int64
	var lastCaptureTime int64

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}

		// Check screen state
		locked, err := session.IsLocked()
		if err != nil {
			log.Warn().Err(err).Msg("failed to check lock state")
		}
		on, _ := session.IsScreenOn()

		emitter.ScreenLocked = locked
		emitter.ScreenOn = on

		if locked || !on {
			log.Debug().Bool("locked", locked).Bool("on", on).Msg("skipping capture")
			continue
		}

		// Capture via screencap
		result, err := screencap.Capture(cfg.Capture.DisplayID, cfg.Capture.Quality)
		if err != nil {
			log.Error().Err(err).Msg("capture failed")
			emitter.LastError = err.Error()
			continue
		}

		timestamp := time.Now().UnixNano()
		if err := buf.Write(timestamp, result.JPEG); err != nil {
			log.Error().Err(err).Msg("failed to write to buffer")
			emitter.LastError = err.Error()
			continue
		}

		captureCount++
		lastCaptureTime = time.Now().UnixMilli()

		emitter.Captures = captureCount
		emitter.LastCaptureMs = lastCaptureTime
		emitter.LastError = ""

		bufCount := buf.Len()
		bufSize := buf.TotalSize()
		emitter.BufferFiles = bufCount
		emitter.BufferBytes = bufSize

		log.Debug().
			Int("w", result.Width).
			Int("h", result.Height).
			Int("bytes", len(result.JPEG)).
			Int("buffered", bufCount).
			Msg("frame captured")
	}
}

// ── test-root ─────────────────────────────────────────────────────────────

var testRootCmd = &cobra.Command{
	Use:   "test-root",
	Short: "Check if root/su is available",
	RunE: func(cmd *cobra.Command, args []string) error {
		if root.CheckRoot() {
			fmt.Println("ROOT_OK")
			return nil
		}
		fmt.Println("ROOT_MISSING")
		os.Exit(1)
		return nil
	},
}

// ── test-screencap ────────────────────────────────────────────────────────

var testScreencapCmd = &cobra.Command{
	Use:   "test-screencap",
	Short: "Capture one screenshot via screencap and save to disk",
	RunE: func(cmd *cobra.Command, args []string) error {
		output, _ := cmd.Flags().GetString("output")
		if output == "" {
			output = "screenshot_test.jpg"
		}

		display, _ := cmd.Flags().GetInt("display")
		quality, _ := cmd.Flags().GetInt("quality")

		if err := root.RequireRoot(true, 5, 30); err != nil {
			return err
		}

		result, err := screencap.Capture(display, quality)
		if err != nil {
			return fmt.Errorf("capture failed: %w", err)
		}

		if err := os.WriteFile(output, result.JPEG, 0644); err != nil {
			return fmt.Errorf("failed to write file: %w", err)
		}

		fmt.Printf("Saved to %s (%dx%d, %d bytes)\n", output, result.Width, result.Height, len(result.JPEG))
		return nil
	},
}

// ── test-lock ─────────────────────────────────────────────────────────────

var testLockCmd = &cobra.Command{
	Use:   "test-lock",
	Short: "Check screen lock state and exit",
	RunE: func(cmd *cobra.Command, args []string) error {
		if err := root.RequireRoot(true, 5, 30); err != nil {
			return err
		}

		locked, err := session.IsLocked()
		if err != nil {
			return fmt.Errorf("failed to check lock state: %w", err)
		}
		on, _ := session.IsScreenOn()

		status := "UNLOCKED"
		if locked {
			status = "LOCKED"
		}
		if !on {
			status += " (SCREEN_OFF)"
		}
		fmt.Println(status)
		return nil
	},
}

// ── status ────────────────────────────────────────────────────────────────

var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Print SERVICE_BRIEFING JSON once and exit",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := config.Load(configPath)
		if err != nil {
			return fmt.Errorf("failed to load config: %w", err)
		}

		rootOK := root.CheckRoot()
		netType := "Unknown"
		if rootOK {
			netType = getNetworkTypeViaSu()
		}

		b := briefing.Briefing{
			Status:      map[bool]string{true: "waiting_root", false: "stopped"}[!rootOK],
			NetworkType: netType,
			ClientName:  cfg.Client.Name,
			DisplayID:   cfg.Capture.DisplayID,
			Timestamp:   time.Now().UnixMilli(),
		}
		jsonBytes, _ := json.Marshal(b)
		fmt.Printf("SERVICE_BRIEFING %s\n", string(jsonBytes))
		return nil
	},
}

func getNetworkTypeViaSu() string {
	out, err := root.RunSu("dumpsys connectivity")
	if err != nil {
		return "Unknown"
	}
	output := string(out)
	if containsAll(output, "WIFI", "CONNECTED") {
		return "WiFi"
	}
	if containsAll(output, "MOBILE", "CONNECTED") {
		return "Cellular"
	}
	if containsAll(output, "ETHERNET", "CONNECTED") {
		return "Ethernet"
	}
	return "None"
}

func containsAll(s string, subs ...string) bool {
	for _, sub := range subs {
		if !strings.Contains(s, sub) {
			return false
		}
	}
	return true
}

func init() {
	testScreencapCmd.Flags().StringP("output", "o", "", "output file path")
	testScreencapCmd.Flags().IntP("display", "d", 0, "display ID to capture")
	testScreencapCmd.Flags().IntP("quality", "q", 85, "JPEG quality (1-100)")
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
