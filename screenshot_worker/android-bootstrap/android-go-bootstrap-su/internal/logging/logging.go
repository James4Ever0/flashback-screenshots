// Package logging sets up zerolog output format and level.
package logging

import (
	"os"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// Setup configures zerolog with the given level and optional log file.
func Setup(level string, logFile string) error {
	lvl, err := zerolog.ParseLevel(level)
	if err != nil {
		lvl = zerolog.InfoLevel
	}
	zerolog.SetGlobalLevel(lvl)
	zerolog.TimeFieldFormat = time.RFC3339Nano

	var writers []zerolog.ConsoleWriter

	// Always log to stdout
	writers = append(writers, zerolog.ConsoleWriter{
		Out:        os.Stdout,
		TimeFormat: time.RFC3339,
		NoColor:    true,
	})

	if logFile != "" {
		f, err := os.OpenFile(logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return err
		}
		writers = append(writers, zerolog.ConsoleWriter{
			Out:        f,
			TimeFormat: time.RFC3339,
			NoColor:    true,
		})
	}

	if len(writers) == 1 {
		log.Logger = zerolog.New(writers[0]).With().Timestamp().Logger()
	} else {
		multi := zerolog.MultiLevelWriter(writers[0], writers[1])
		log.Logger = zerolog.New(multi).With().Timestamp().Logger()
	}

	return nil
}
