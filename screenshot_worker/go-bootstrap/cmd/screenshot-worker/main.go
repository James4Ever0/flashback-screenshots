package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/kbinani/screenshot"
	"github.com/rs/zerolog/log"
	"github.com/spf13/cobra"

	"screenshot-worker/internal/buffer"
	"screenshot-worker/internal/capture"
	"screenshot-worker/internal/config"
	"screenshot-worker/internal/logging"
	"screenshot-worker/internal/service"
	"screenshot-worker/internal/session"
	"screenshot-worker/internal/transport"
)

var (
	configPath string
	rootCmd    = &cobra.Command{
		Use:   "screenshot-worker",
		Short: "Screenshot worker client",
	}
)

func init() {
	rootCmd.PersistentFlags().StringVarP(&configPath, "config", "c", "", "config file path")

	rootCmd.AddCommand(
		daemonCmd,
		testLockCmd,
		testCaptureCmd,
		testWindowCmd,
		installCmd,
		uninstallCmd,
		statusCmd,
	)
}

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

		log.Info().Str("name", cfg.Client.Name).Str("server", cfg.Server.URL).Msg("starting daemon")

		// Session monitor
		mon := session.NewMonitor()

		// Disk buffer
		bufDir := cfg.Buffer.Path
		if bufDir == "" {
			bufDir = filepath.Join(defaultDataDir(), "buffer")
		}
		buf, err := buffer.New(bufDir, cfg.Buffer.MaxFiles)
		if err != nil {
			return fmt.Errorf("failed to create buffer: %w", err)
		}
		log.Info().Str("dir", bufDir).Int("max_files", cfg.Buffer.MaxFiles).Msg("buffer ready")

		// Transport
		client := transport.NewClient(cfg.Server.URL, cfg.Client.Name, buf)

		// Context for graceful shutdown
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		// Start capture loop
		go captureLoop(ctx, cfg, mon, buf)

		// Start transport
		go func() {
			if err := client.Run(ctx); err != nil {
				log.Error().Err(err).Msg("transport exited")
			}
		}()

		// Wait for interrupt
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh

		log.Info().Msg("shutting down")
		cancel()
		client.Stop()
		time.Sleep(100 * time.Millisecond)
		return nil
	},
}

func captureLoop(ctx context.Context, cfg config.Config, mon session.Monitor, buf *buffer.Buffer) {
	ticker := time.NewTicker(time.Duration(cfg.Capture.IntervalSec) * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}

		locked, err := mon.IsLocked()
		if err != nil {
			log.Warn().Err(err).Msg("failed to check lock state")
			continue
		}
		if locked {
			log.Debug().Msg("screen locked, skipping capture")
			continue
		}

		frame, err := capture.Capture(0, cfg.Capture.Quality, locked)
		if err != nil {
			log.Error().Err(err).Msg("capture failed")
			continue
		}

		if err := buf.Write(frame.Timestamp, frame.ImageJPEG); err != nil {
			log.Error().Err(err).Msg("failed to write to buffer")
			continue
		}

		count, _ := buf.Len()
		log.Debug().Int("buffered", count).Msg("frame captured")
	}
}

var testLockCmd = &cobra.Command{
	Use:   "test-lock",
	Short: "Check screen lock state and exit",
	RunE: func(cmd *cobra.Command, args []string) error {
		mon := session.NewMonitor()
		locked, err := mon.IsLocked()
		if err != nil {
			return fmt.Errorf("failed to check lock state: %w", err)
		}
		if locked {
			fmt.Println("LOCKED")
			os.Exit(0)
		}
		fmt.Println("UNLOCKED")
		return nil
	},
}

var testCaptureCmd = &cobra.Command{
	Use:   "test-capture",
	Short: "Capture one screenshot and save to disk",
	RunE: func(cmd *cobra.Command, args []string) error {
		output, _ := cmd.Flags().GetString("output")
		if output == "" {
			output = "screenshot_test.jpg"
		}

		mon := session.NewMonitor()
		locked, _ := mon.IsLocked()

		frame, err := capture.Capture(0, 85, locked)
		if err != nil {
			return fmt.Errorf("capture failed: %w", err)
		}

		if err := os.WriteFile(output, frame.ImageJPEG, 0644); err != nil {
			return fmt.Errorf("failed to write file: %w", err)
		}

		fmt.Printf("Saved to %s (%dx%d, %d bytes)\n", output, frame.Width, frame.Height, len(frame.ImageJPEG))
		return nil
	},
}

var testWindowCmd = &cobra.Command{
	Use:   "test-window",
	Short: "Print display information",
	RunE: func(cmd *cobra.Command, args []string) error {
		n := screenshot.NumActiveDisplays()
		fmt.Printf("Active displays: %d\n", n)
		for i := 0; i < n; i++ {
			bounds := screenshot.GetDisplayBounds(i)
			fmt.Printf("  Display %d: %dx%d at (%d,%d)\n", i, bounds.Dx(), bounds.Dy(), bounds.Min.X, bounds.Min.Y)
		}
		return nil
	},
}

var installCmd = &cobra.Command{
	Use:   "install",
	Short: "Install as a background service",
	RunE: func(cmd *cobra.Command, args []string) error {
		system, _ := cmd.Flags().GetBool("system")
		return service.New().Install(system)
	},
}

var uninstallCmd = &cobra.Command{
	Use:   "uninstall",
	Short: "Remove the background service",
	RunE: func(cmd *cobra.Command, args []string) error {
		return service.New().Uninstall()
	},
}

var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Check service status",
	RunE: func(cmd *cobra.Command, args []string) error {
		status, err := service.New().Status()
		if err != nil {
			return err
		}
		fmt.Println(status)
		return nil
	},
}

func init() {
	testCaptureCmd.Flags().StringP("output", "o", "", "output file path")
	installCmd.Flags().Bool("system", false, "install as system service (requires admin)")
}

func defaultDataDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return filepath.Join(home, ".local", "share", "screenshot-worker")
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
