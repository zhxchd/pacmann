//go:build !unix

package pianopir

import "time"

func cpuNow() time.Duration {
	return 0
}
