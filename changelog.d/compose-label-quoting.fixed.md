Deployment: the generated labels override quotes a value that YAML 1.2 reads
as a number, such as the repo-id digest `096780950e46`, so compose labels the
containers with the string and `osprey reset` still finds them.
