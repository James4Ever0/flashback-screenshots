// Package config handles TOML configuration loading and validation.
package config

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

// Config holds all application configuration.
type Config struct {
	Server   ServerConfig   `toml:"server"`
	Capture  CaptureConfig  `toml:"capture"`
	Buffer   BufferConfig   `toml:"buffer"`
	Client   ClientConfig   `toml:"client"`
	Root     RootConfig     `toml:"root"`
	Briefing BriefingConfig `toml:"briefing"`
	Upload   UploadConfig   `toml:"upload"`
}

// ServerConfig holds server connection settings.
type ServerConfig struct {
	URL string `toml:"url"`
}

// CaptureConfig holds screenshot capture settings.
type CaptureConfig struct {
	IntervalSec int `toml:"interval_sec"`
	Quality     int `toml:"quality"`
	DisplayID   int `toml:"display_id"`
}

// BufferConfig holds disk ring buffer settings.
type BufferConfig struct {
	MaxFiles int    `toml:"max_files"`
	Path     string `toml:"path"`
}

// ClientConfig holds client identification and logging settings.
type ClientConfig struct {
	Name     string `toml:"name"`
	LogLevel string `toml:"log_level"`
	LogFile  string `toml:"log_file"`
}

// RootConfig holds root/su permission settings.
type RootConfig struct {
	AutoRequest      bool `toml:"auto_request"`
	RetryIntervalSec int  `toml:"retry_interval_sec"`
	MaxRetrySec      int  `toml:"max_retry_sec"`
}

// BriefingConfig holds SERVICE_BRIEFING output settings.
type BriefingConfig struct {
	IntervalSec int `toml:"interval_sec"`
}

// UploadConfig holds upload restriction settings.
type UploadConfig struct {
	WifiOnly bool `toml:"wifi_only"`
}

// Default returns a Config with sensible defaults.
func Default() Config {
	return Config{
		Server: ServerConfig{
			URL: "ws://localhost:8080/ws",
		},
		Capture: CaptureConfig{
			IntervalSec: 5,
			Quality:     80,
			DisplayID:   0,
		},
		Buffer: BufferConfig{
			MaxFiles: 1000,
			Path:     "/data/local/tmp/go-screenshot-buffer",
		},
		Client: ClientConfig{
			Name:     "android-device",
			LogLevel: "info",
		},
		Root: RootConfig{
			AutoRequest:      true,
			RetryIntervalSec: 10,
			MaxRetrySec:      60,
		},
		Briefing: BriefingConfig{
			IntervalSec: 3,
		},
		Upload: UploadConfig{
			WifiOnly: true,
		},
	}
}

// Load reads configuration from a TOML file.
func Load(path string) (Config, error) {
	cfg := Default()

	if path == "" {
		path = findConfigFile()
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		return cfg, fmt.Errorf("config file not found: %s", path)
	}

	if _, err := toml.DecodeFile(path, &cfg); err != nil {
		return cfg, fmt.Errorf("failed to decode config: %w", err)
	}

	if err := cfg.Validate(); err != nil {
		return cfg, err
	}

	return cfg, nil
}

// Validate checks that the configuration is sane.
func (c Config) Validate() error {
	if c.Server.URL == "" {
		return fmt.Errorf("server.url is required")
	}
	if c.Capture.IntervalSec < 1 {
		return fmt.Errorf("capture.interval_sec must be >= 1")
	}
	if c.Capture.Quality < 1 || c.Capture.Quality > 100 {
		return fmt.Errorf("capture.quality must be between 1 and 100")
	}
	if c.Buffer.MaxFiles < 1 {
		return fmt.Errorf("buffer.max_files must be >= 1")
	}
	if c.Client.Name == "" {
		return fmt.Errorf("client.name is required")
	}
	if c.Buffer.Path == "" {
		c.Buffer.Path = "/data/local/tmp/go-screenshot-buffer"
	}
	return nil
}

// findConfigFile searches common config locations.
func findConfigFile() string {
	candidates := []string{
		"config.toml",
	}

	if home, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates,
			filepath.Join(home, ".config", "screenshot-worker", "config.toml"),
		)
	}

	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	return "config.toml"
}
