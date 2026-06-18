//go:build windows

package session

import (
	"sync/atomic"
	"syscall"
	"unsafe"
)

var (
	wtsapi32                = syscall.NewLazyDLL("wtsapi32.dll")
	procWTSRegister         = wtsapi32.NewProc("WTSRegisterSessionNotification")
	procWTSUnRegister       = wtsapi32.NewProc("WTSUnRegisterSessionNotification")

	user32                  = syscall.NewLazyDLL("user32.dll")
	procCreateWindowEx      = user32.NewProc("CreateWindowExW")
	procDefWindowProc       = user32.NewProc("DefWindowProcW")
	procRegisterClassEx     = user32.NewProc("RegisterClassExW")
	procGetMessage          = user32.NewProc("GetMessageW")
	procDispatchMessage     = user32.NewProc("DispatchMessageW")
	procTranslateMessage    = user32.NewProc("TranslateMessage")
	procPostQuitMessage     = user32.NewProc("PostQuitMessage")
	procPeekMessage         = user32.NewProc("PeekMessageW")

	locked int32 = 0
)

const (
	NOTIFY_FOR_THIS_SESSION = 0
	WM_WTSSESSION_CHANGE    = 0x02B1
	WTS_SESSION_LOCK        = 0x7
	WTS_SESSION_UNLOCK      = 0x8
)

// windowsMonitor tracks session lock via WTSRegisterSessionNotification.
type windowsMonitor struct {
	hwnd syscall.Handle
}

// NewMonitor creates a session lock monitor for Windows.
func NewMonitor() Monitor {
	m := &windowsMonitor{}
	m.hwnd = m.createMessageWindow()
	if m.hwnd != 0 {
		procWTSRegister.Call(uintptr(m.hwnd), NOTIFY_FOR_THIS_SESSION)
		go m.messageLoop()
	}
	return m
}

func (m *windowsMonitor) IsLocked() (bool, error) {
	return atomic.LoadInt32(&locked) != 0, nil
}

type WNDCLASSEX struct {
	CbSize        uint32
	Style         uint32
	LpfnWndProc   uintptr
	CbClsExtra    int32
	CbWndExtra    int32
	HInstance     syscall.Handle
	HIcon         syscall.Handle
	HCursor       syscall.Handle
	HbrBackground syscall.Handle
	LpszMenuName  *uint16
	LpszClassName *uint16
	HIconSm       syscall.Handle
}

type MSG struct {
	Hwnd    syscall.Handle
	Message uint32
	WParam  uintptr
	LParam  uintptr
	Time    uint32
	Pt      struct{ X, Y int32 }
}

func (m *windowsMonitor) createMessageWindow() syscall.Handle {
	className, _ := syscall.UTF16PtrFromString("ScreenshotWorkerSessionMonitor")

	wc := WNDCLASSEX{
		CbSize:        uint32(unsafe.Sizeof(WNDCLASSEX{})),
		LpfnWndProc:   syscall.NewCallback(m.windowProc),
		LpszClassName: className,
	}

	ret, _, _ := procRegisterClassEx.Call(uintptr(unsafe.Pointer(&wc)))
	if ret == 0 {
		return 0
	}

	hwnd, _, _ := procCreateWindowEx.Call(
		0,
		uintptr(unsafe.Pointer(className)),
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	)
	return syscall.Handle(hwnd)
}

func (m *windowsMonitor) windowProc(hwnd syscall.Handle, msg uint32, wParam uintptr, lParam uintptr) uintptr {
	if msg == WM_WTSSESSION_CHANGE {
		switch wParam {
		case WTS_SESSION_LOCK:
			atomic.StoreInt32(&locked, 1)
		case WTS_SESSION_UNLOCK:
			atomic.StoreInt32(&locked, 0)
		}
		return 0
	}
	ret, _, _ := procDefWindowProc.Call(uintptr(hwnd), uintptr(msg), wParam, lParam)
	return ret
}

func (m *windowsMonitor) messageLoop() {
	var msg MSG
	for {
		ret, _, _ := procGetMessage.Call(uintptr(unsafe.Pointer(&msg)), 0, 0, 0)
		if ret == 0 {
			break
		}
		procTranslateMessage.Call(uintptr(unsafe.Pointer(&msg)))
		procDispatchMessage.Call(uintptr(unsafe.Pointer(&msg)))
	}
}
