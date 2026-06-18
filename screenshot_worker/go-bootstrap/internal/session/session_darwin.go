//go:build darwin

package session

/*
#include <CoreFoundation/CoreFoundation.h>

// CGSessionCopyCurrentDictionary is available in CoreGraphics
extern CFDictionaryRef CGSessionCopyCurrentDictionary(void);
*/
import "C"
import "unsafe"

// darwinMonitor checks session lock via CoreGraphics.
type darwinMonitor struct{}

// NewMonitor creates a session lock monitor for macOS.
func NewMonitor() Monitor {
	return &darwinMonitor{}
}

// IsLocked reads CGSessionScreenIsLocked from CGSessionCopyCurrentDictionary.
func (m *darwinMonitor) IsLocked() (bool, error) {
	dict := C.CGSessionCopyCurrentDictionary()
	if dict == 0 {
		return false, nil
	}
	defer C.CFRelease(C.CFTypeRef(dict))

	key := C.CFStringCreateWithCString(0, (*C.char)(unsafe.Pointer(&[]byte("CGSessionScreenIsLocked")[0])), C.kCFStringEncodingUTF8)
	defer C.CFRelease(C.CFTypeRef(key))

	val := C.CFDictionaryGetValue(dict, unsafe.Pointer(key))
	if val == nil {
		return false, nil
	}

	return CFBooleanToBool(C.CFBooleanRef(val)), nil
}

func CFBooleanToBool(b C.CFBooleanRef) bool {
	return C.CFBooleanGetValue(b) != 0
}
