package ifneeded Tcldot 2.28.0 "
	load [file join $dir libtcldot.0.dylib] Tcldot"
package ifneeded Tclpathplan 2.28.0 "
	load [file join $dir libtclplan.0.dylib] Tclpathplan"
package ifneeded Gdtclft 2.28.0 "
	load [file join $dir libgdtclft.0.dylib] Gdtclft"
package ifneeded gv 0 "
	load [file join $dir libgv_tcl.so] gv"
package ifneeded Tkspline 2.28.0 "
	package require Tk 8.3
	load [file join $dir libtkspline.0.dylib] Tkspline"
# end
