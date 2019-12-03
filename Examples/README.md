# Examples for MVMO test

## Testing built-in MVMO functions
The file `mvmo_test_functions.py` contains example to show usage of MVMO for built-in standard test functions.

## MVMO for Optimal Power Flow using pandapower
The example in file `pandapower_opf_with_mvmo.py` shows how MVMO can be used for optimal power flow calculation using constraints as function definition in MVMO. Network is defined as a [pandapower](http://www.pandapower.org/references/) net. An IEEE 9 bus system is used. Note: OPF costs from pandapower are negative, as per its convention to reduce the total generation. In MVMO, the costs are shown as positive, while it is still minimizing the total generation. 