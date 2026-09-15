Readonly Python runs can read through `osprey.runtime` again on a pyepics
deployment. The readonly guard refused every shared-library load, including
the one pyepics makes to reach Channel Access; that load is now permitted, and
the library's put entry points are refused on the handle it gets.
