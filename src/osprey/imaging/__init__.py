"""Picture handling shared by ARIEL and the render worker.

This package stays import-light: the render worker imports
``osprey.imaging.formats`` in an isolated interpreter and must not pull in any
service code, so nothing is re-exported here.
"""
