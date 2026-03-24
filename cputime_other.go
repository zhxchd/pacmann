//go:build !unix

package main

import "time"

func cpuNow() time.Duration {
	return 0
}
