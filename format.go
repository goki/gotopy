// Copyright 2020 The Go-Python Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is based on gofmt source code:

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"go/ast"
	"go/token"

	"github.com/goki/gotopy/pyprint"
)

// format formats the given package file originally obtained from src
// and adjusts the result based on the original source via sourceAdj
// and indentAdj.
func format(
	fset *token.FileSet,
	file *ast.File,
	sourceAdj func(src []byte, indent int) []byte,
	indentAdj int,
	src []byte,
	cfg pyprint.Config,
) ([]byte, error) {
	if sourceAdj == nil {
		// Complete source file.
		var buf bytes.Buffer
		err := cfg.Fprint(&buf, fset, file)
		if err != nil {
			return nil, err
		}
		pyfix := pyEdits(buf.Bytes(), true)
		return pyfix, nil
		// return buf.Bytes(), nil
	}

	// Partial source file.
	// Determine and prepend leading space.
	i, j := 0, 0
	for j < len(src) && isSpace(src[j]) {
		if src[j] == '\n' {
			i = j + 1 // byte offset of last line in leading space
		}
		j++
	}
	var res []byte
	res = append(res, src[:i]...)

	// Determine and prepend indentation of first code line.
	// Spaces are ignored unless there are no tabs,
	// in which case spaces count as one tab.
	indent := 0
	hasSpace := false
	for _, b := range src[i:j] {
		switch b {
		case ' ':
			hasSpace = true
		case '\t':
			indent++
		}
	}
	if indent == 0 && hasSpace {
		indent = 1
	}
	for i := 0; i < indent; i++ {
		res = append(res, '\t')
	}

	// Format the source.
	// Write it without any leading and trailing space.
	cfg.Indent = indent + indentAdj
	var buf bytes.Buffer
	err := cfg.Fprint(&buf, fset, file)
	if err != nil {
		return nil, err
	}

	pyfix := pyEdits(buf.Bytes(), true)

	out := sourceAdj(pyfix, cfg.Indent)

	// If the adjusted output is empty, the source
	// was empty but (possibly) for white space.
	// The result is the incoming source.
	if len(out) == 0 {
		return src, nil
	}

	// Otherwise, append output to leading space.
	res = append(res, out...)

	// Determine and append trailing space.
	i = len(src)
	for i > 0 && isSpace(src[i-1]) {
		i--
	}
	return append(res, src[i:]...), nil
}

// isSpace reports whether the byte is a space character.
// isSpace defines a space as being among the following bytes: ' ', '\t', '\n' and '\r'.
func isSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}
