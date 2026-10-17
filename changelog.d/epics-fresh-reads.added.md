`control_system.connector.<type>.fresh_reads` is accepted by EPICS-family
connector blocks. Every Channel Access read already asks the IOC rather than a
monitor cache, so an IOC that computes its readbacks on get is read fresh
whatever the key says; the key is kept, and validated, so a configuration that
sets it per block (a stand-in simulator but not the live machine) keeps loading.
