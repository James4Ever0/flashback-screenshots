// Package transport handles WebSocket connection to the server.
package transport

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"net/url"
	"time"

	"android-go-bootstrap-su/internal/netcheck"
	"github.com/gorilla/websocket"
	"github.com/rs/zerolog/log"
)

// FrameSender abstracts the buffer for reading/removing frames.
type FrameSender interface {
	Oldest() (string, []byte, error)
	Remove(string) error
}

// Client manages the WebSocket connection.
type Client struct {
	serverURL    string
	clientName   string
	sender       FrameSender
	wifiOnly     bool
	conn         *websocket.Conn
	dialer       websocket.Dialer
	stop         chan struct{}
	framesSent   int64
	lastSendTime int64
}

// ClientHello is the first message sent after connecting.
type ClientHello struct {
	Type    string `json:"type"`
	Name    string `json:"name"`
	Version string `json:"version"`
}

// FrameMessage wraps a screenshot frame for transmission.
type FrameMessage struct {
	Type      string `json:"type"`
	Timestamp int64  `json:"timestamp"`
	Filename  string `json:"filename"`
	Data      string `json:"data"`
}

// ServerCommand represents a command from the server.
type ServerCommand struct {
	Type string `json:"type"`
}

// NewClient creates a new transport client.
func NewClient(serverURL, clientName string, sender FrameSender, wifiOnly bool) *Client {
	return &Client{
		serverURL:  serverURL,
		clientName: clientName,
		sender:     sender,
		wifiOnly:   wifiOnly,
		dialer: websocket.Dialer{
			HandshakeTimeout: 10 * time.Second,
		},
		stop: make(chan struct{}),
	}
}

// FramesSent returns total frames uploaded this session.
func (c *Client) FramesSent() int64 { return c.framesSent }

// LastSendTime returns Unix millis of last successful upload.
func (c *Client) LastSendTime() int64 { return c.lastSendTime }

// Connected returns true if the websocket is currently connected.
func (c *Client) Connected() bool {
	return c.conn != nil
}

// Run connects to the server and manages the connection lifecycle.
func (c *Client) Run(ctx context.Context) error {
	backoff := time.Second
	maxBackoff := 60 * time.Second

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-c.stop:
			return nil
		default:
		}

		if err := c.connectAndRun(ctx); err != nil {
			log.Error().Err(err).Dur("backoff", backoff).Msg("connection failed, retrying")
		}

		time.Sleep(backoff + time.Duration(rand.Int63n(int64(backoff)/2)))
		if backoff < maxBackoff {
			backoff *= 2
		}
	}
}

// Stop signals the client to disconnect and stop retrying.
func (c *Client) Stop() {
	close(c.stop)
	if c.conn != nil {
		c.conn.Close()
	}
}

func (c *Client) connectAndRun(ctx context.Context) error {
	u, err := url.Parse(c.serverURL)
	if err != nil {
		return fmt.Errorf("invalid server URL: %w", err)
	}

	q := u.Query()
	q.Set("name", c.clientName)
	u.RawQuery = q.Encode()

	headers := http.Header{}
	conn, resp, err := c.dialer.Dial(u.String(), headers)
	if err != nil {
		if resp != nil {
			return fmt.Errorf("dial failed (status %d): %w", resp.StatusCode, err)
		}
		return fmt.Errorf("dial failed: %w", err)
	}
	c.conn = conn
	defer conn.Close()

	log.Info().Str("url", c.serverURL).Msg("connected to server")

	hello := ClientHello{Type: "hello", Name: c.clientName, Version: "0.1.0"}
	if err := conn.WriteJSON(hello); err != nil {
		return fmt.Errorf("failed to send hello: %w", err)
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	errCh := make(chan error, 2)
	go func() { errCh <- c.readLoop(ctx) }()
	go func() { errCh <- c.writeLoop(ctx) }()

	err = <-errCh
	cancel()
	<-errCh
	return err
}

func (c *Client) readLoop(ctx context.Context) error {
	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}

		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		_, data, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				log.Info().Msg("connection closed by server")
				return nil
			}
			return fmt.Errorf("read error: %w", err)
		}

		var cmd ServerCommand
		if err := json.Unmarshal(data, &cmd); err != nil {
			log.Warn().Err(err).Msg("failed to unmarshal server command")
			continue
		}

		switch cmd.Type {
		case "ping":
			if err := c.conn.WriteJSON(ServerCommand{Type: "pong"}); err != nil {
				return fmt.Errorf("failed to send pong: %w", err)
			}
		case "ack":
			log.Debug().Msg("received ack")
		}
	}
}

func (c *Client) writeLoop(ctx context.Context) error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil
		case <-c.stop:
			return nil
		case <-ticker.C:
		}

		if c.wifiOnly && !netcheck.IsWiFi() {
			continue
		}

		filename, data, err := c.sender.Oldest()
		if err != nil {
			log.Error().Err(err).Msg("failed to read from buffer")
			continue
		}
		if filename == "" {
			continue
		}

		msg := FrameMessage{
			Type:      "frame",
			Timestamp: time.Now().Unix(),
			Filename:  filename,
			Data:      base64.StdEncoding.EncodeToString(data),
		}

		if err := c.conn.WriteJSON(msg); err != nil {
			return fmt.Errorf("failed to send frame: %w", err)
		}

		if err := c.sender.Remove(filename); err != nil {
			log.Warn().Err(err).Str("file", filename).Msg("failed to remove frame from buffer")
		}

		c.framesSent++
		c.lastSendTime = time.Now().UnixMilli()
		log.Debug().Str("file", filename).Int("size", len(data)).Msg("sent frame")
	}
}
