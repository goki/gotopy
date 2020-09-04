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
	"go/parser"
	"go/token"
	"strings"
)

// parse parses src, which was read from the named file,
// as a Go source file, declaration, or statement list.
func parse(fset *token.FileSet, filename string, src []byte, fragmentOk bool) (
	file *ast.File,
	sourceAdj func(src []byte, indent int) []byte,
	indentAdj int,
	err error,
) {
	// Try as whole source file.
	file, err = parser.ParseFile(fset, filename, src, parserMode)
	// If there's no error, return. If the error is that the source file didn't begin with a
	// package line and source fragments are ok, fall through to
	// try as a source fragment. Stop and return on any other error.
	if err == nil || !fragmentOk || !strings.Contains(err.Error(), "expected 'package'") {
		return
	}

	// If this is a declaration list, make it a source file
	// by inserting a package clause.
	// Insert using a ';', not a newline, so that the line numbers
	// in psrc match the ones in src.
	psrc := append([]byte("package p;"), src...)
	file, err = parser.ParseFile(fset, filename, psrc, parserMode)
	if err == nil {
		sourceAdj = func(src []byte, indent int) []byte {
			// Remove the package clause.
			// Gofmt has turned the ';' into a '\n'.
			src = src[indent+len("package p\n"):]
			return bytes.TrimSpace(src)
		}
		return
	}
	// If the error is that the source file didn't begin with a
	// declaration, fall through to try as a statement list.
	// Stop and return on any other error.
	if !strings.Contains(err.Error(), "expected declaration") {
		return
	}

	// If this is a statement list, make it a source file
	// by inserting a package clause and turning the list
	// into a function body. This handles expressions too.
	// Insert using a ';', not a newline, so that the line numbers
	// in fsrc match the ones in src. Add an extra '\n' before the '}'
	// to make sure comments are flushed before the '}'.
	fsrc := append(append([]byte("package p; func _() {"), src...), '\n', '\n', '}')
	file, err = parser.ParseFile(fset, filename, fsrc, parserMode)
	if err == nil {
		sourceAdj = func(src []byte, indent int) []byte {
			// Cap adjusted indent to zero.
			if indent < 0 {
				indent = 0
			}
			// Remove the wrapping.
			// Gofmt has turned the "; " into a "\n\n".
			// There will be two non-blank lines with indent, hence 2*indent.
			src = src[2*indent+len("package p\n\nfunc _() {"):]
			// Remove only the "}\n" suffix: remaining whitespaces will be trimmed anyway
			src = src[:len(src)-len("}\n")]
			return bytes.TrimSpace(src)
		}
		// Gofmt has also indented the function body one level.
		// Adjust that with indentAdj.
		indentAdj = -1
	}

	// Succeeded, or out of options.
	return
}
