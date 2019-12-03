## MVMO for Optimal Power Flow using pandapower
The example shows how MVMO can be used for optimal power flow calculation using constraints as function definition in MVMO. Network is defined as a [pandapower](http://www.pandapower.org/references/) net
IEEE 9 bus system is used.
Note: OPF costs from pandapower are negative, as per its convention to reduce the total generation. In MVMO, the costs are shown as positive, while it is still minimizing the total generation. 