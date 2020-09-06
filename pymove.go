// Copyright 2020 The Go-Python Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"strings"
)

// moveLines moves the st,ed region to 'to' line
func moveLines(lines *[][]byte, to, st, ed int) {
	mvln := (*lines)[st:ed]
	btwn := (*lines)[to:st]
	aft := (*lines)[ed:len(*lines)]
	nln := make([][]byte, to, len(*lines))
	copy(nln, (*lines)[:to])
	nln = append(nln, mvln...)
	nln = append(nln, btwn...)
	nln = append(nln, aft...)
	*lines = nln
}

// pyMove moves python segments around, e.g., methods
// into their proper classes
func pyMove(src []byte) []byte {
	type sted struct {
		st, ed int
	}

	classes := map[string]sted{}

	nl := []byte("\n")
	class := []byte("class ")
	pymark := []byte("<<<<")
	pyend := []byte(">>>>")
	fmtPrintf := []byte("fmt.Printf")
	fmtSprintf := []byte("fmt.Sprintf(")
	prints := []byte("print")
	endclass := "EndClass: "
	method := "Method: "
	endmethod := "EndMethod"

	lines := bytes.Split(src, nl)

	lastMethSt := -1
	var lastMeth string
	curComSt := -1
	lastComSt := -1
	lastComEd := -1

	li := 0
	for {
		if li >= len(lines) {
			break
		}
		ln := lines[li]
		if len(ln) > 0 && ln[0] == '#' {
			if curComSt >= 0 {
				lastComEd = li
			} else {
				curComSt = li
				lastComSt = li
				lastComEd = li
			}
		} else {
			curComSt = -1
		}

		switch {
		case bytes.Equal(ln, []byte("	:")) || bytes.Equal(ln, []byte(":")):
			lines = append(lines[:li], lines[li+1:]...) // delete marker
			li--
		case bytes.HasPrefix(ln, class):
			cl := string(ln[len(class):])
			if idx := strings.Index(cl, "("); idx > 0 {
				cl = cl[:idx]
			} else if idx := strings.Index(cl, ":"); idx > 0 { // should have
				cl = cl[:idx]
			}
			cl = strings.TrimSpace(cl)
			classes[cl] = sted{st: li}
			// fmt.Printf("cl: %s at %d\n", cl, li)
		case bytes.HasPrefix(ln, pymark) && bytes.HasSuffix(ln, pyend):
			tag := string(ln[4 : len(ln)-4])
			// fmt.Printf("tag: %s at: %d\n", tag, li)
			switch {
			case strings.HasPrefix(tag, endclass):
				cl := tag[len(endclass):]
				st := classes[cl]
				classes[cl] = sted{st: st.st, ed: li}
				lines = append(lines[:li], lines[li+1:]...) // delete marker
				// fmt.Printf("cl: %s at %v\n", cl, classes[cl])
				li--
			case strings.HasPrefix(tag, method):
				cl := tag[len(method):]
				lines = append(lines[:li], lines[li+1:]...) // delete marker
				li--
				lastMeth = cl
				if lastComEd == li {
					lines = append(lines[:lastComSt], lines[lastComEd+1:]...) // delete comments
					lastMethSt = lastComSt
					li = lastComSt - 1
				} else {
					lastMethSt = li + 1
				}
			case tag == endmethod:
				se, ok := classes[lastMeth]
				if ok {
					lines = append(lines[:li], lines[li+1:]...) // delete marker
					moveLines(&lines, se.ed, lastMethSt, li+1)  // extra blank
					classes[lastMeth] = sted{st: se.st, ed: se.ed + ((li + 1) - lastMethSt)}
					li -= 2
				}
			}
		case bytes.Contains(ln, fmtSprintf):
			if bytes.Contains(ln, []byte("%")) {
				ln = bytes.Replace(ln, []byte(`", `), []byte(`" % (`), -1)
			}
			ln = bytes.Replace(ln, fmtSprintf, []byte{}, -1)
			lines[li] = ln
		case bytes.Contains(ln, fmtPrintf):
			if bytes.Contains(ln, []byte("%")) {
				ln = bytes.Replace(ln, []byte(`", `), []byte(`" % `), -1)
			}
			ln = bytes.Replace(ln, fmtPrintf, prints, -1)
			lines[li] = ln
		}
		li++
	}

	return bytes.Join(lines, nl)
}
