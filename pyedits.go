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

// pyEdits performs post-generation edits for python
// * moves python segments around, e.g., methods
// into their proper classes
// * fixes printf, slice other common code
func pyEdits(src []byte, gopy bool) []byte {
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
	eqappend := []byte("= append(")
	itoa := []byte("strconv.Itoa")
	float64p := []byte("float64(")
	float32p := []byte("float32(")
	floatp := []byte("float(")
	slicestr := []byte("[]string(")
	sliceint := []byte("[]int(")
	slicefloat64 := []byte("[]float64(")
	slicefloat32 := []byte("[]float32(")
	goslicestr := []byte("go.Slice_string([")
	gosliceint := []byte("go.Slice_int([")
	goslicefloat64 := []byte("go.Slice_float64([")
	goslicefloat32 := []byte("go.Slice_float32([")

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

		ln = bytes.Replace(ln, float64p, floatp, -1)
		ln = bytes.Replace(ln, float32p, floatp, -1)
		lines[li] = ln

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
		case bytes.Contains(ln, eqappend):
			idx := bytes.Index(ln, eqappend)
			comi := bytes.Index(ln[idx+len(eqappend):], []byte(","))
			nln := make([]byte, idx-1)
			copy(nln, ln[:idx-1])
			nln = append(nln, []byte(".append(")...)
			nln = append(nln, ln[idx+len(eqappend)+comi+1:]...)
			lines[li] = nln
		case bytes.Contains(ln, slicestr):
			ln = bytes.Replace(ln, slicestr, goslicestr, -1)
			ln = bytes.Replace(ln, []byte(")"), []byte("])"), 1)
			lines[li] = ln
		case bytes.Contains(ln, sliceint):
			ln = bytes.Replace(ln, sliceint, gosliceint, -1)
			ln = bytes.Replace(ln, []byte(")"), []byte("])"), 1)
			lines[li] = ln
		case bytes.Contains(ln, slicefloat64):
			ln = bytes.Replace(ln, slicefloat64, goslicefloat64, -1)
			ln = bytes.Replace(ln, []byte(")"), []byte("])"), 1)
			lines[li] = ln
		case bytes.Contains(ln, slicefloat32):
			ln = bytes.Replace(ln, slicefloat32, goslicefloat32, -1)
			ln = bytes.Replace(ln, []byte(")"), []byte("])"), 1)
			lines[li] = ln
		case bytes.Contains(ln, itoa):
			ln = bytes.Replace(ln, itoa, []byte(`str`), -1)
			lines[li] = ln
		}
		li++
	}

	return bytes.Join(lines, nl)
}
