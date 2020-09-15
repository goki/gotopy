# GoToPy

GoToPy is a Go to Python converter -- translates Go code into Python code.

It is based on the Go `gofmt` command source code and the go `printer` package, which parses Go files and writes them out according to standard go formatting.

We have modified the `printer` code in the `pyprint` package to instead print out Python code.

The `-gopy` flag generates [GoPy](https:://github.com/go-python/gopy) specific Python code, including:

* `nil` -> `go.nil`
* `[]string{...}` -> `go.Slice_string([...])`  etc for int, float64, float32

The `-gogi` flag generates [GoGi](https:://github.com/goki/gi) specific Python code, including:

* struct tags generate: `self.SetTags()` call, for the `pygiv.ClassViewObj` class, which then provides an automatic GUI view with tag-based formatting of struct fields.

# TODO

* switch -> ifs.. -- grab switch expr and put into each if


