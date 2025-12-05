"""Configuration Parameter Loading and Management System.

This module provides a sophisticated parameter loading system that supports YAML
configuration files with import directives, hierarchical parameter access, and
environment variable expansion. The system is designed for flexible deployment
configurations where complex service arrangements need robust parameter management.

The module implements a type-safe parameter access pattern through the Params class,
which provides dot-notation access to nested configuration data while maintaining
validation and error handling. Invalid parameter access returns InvalidParam objects
rather than raising exceptions, enabling graceful degradation in configuration scenarios.

Key Features:
    - Recursive YAML file imports with circular dependency detection
    - Environment variable expansion in string values
    - Type-safe parameter access with validation
    - Graceful error handling for missing configuration keys
    - Deep dictionary merging for configuration composition

Examples:
    Basic parameter loading::

        >>> params = load_params('config.yml')
        >>> database_host = params.database.host
        >>> timeout = params.services.timeout

    Configuration with imports::

        # base_config.yml
        database:
          host: localhost
          port: 5432

        # app_config.yml
        import: base_config.yml
        database:
          name: myapp  # Merged with base config
        services:
          timeout: 30

        >>> params = load_params('app_config.yml')
        >>> print(params.database.host)  # 'localhost' from base
        >>> print(params.database.name)  # 'myapp' from app config

    Environment variable expansion::

        # config.yml
        project_root: ${PROJECT_ROOT}
        data_dir: ${PROJECT_ROOT}/data

        >>> os.environ['PROJECT_ROOT'] = '/home/user/project'
        >>> params = load_params('config.yml')
        >>> print(params.data_dir)  # '/home/user/project/data'

.. seealso::
   :class:`Params` : Main parameter container with hierarchical access
   :class:`InvalidParam` : Error handling for missing configuration keys
   :func:`_load_yaml` : Core YAML loading with import processing
   :mod:`deployment.container_manager` : Uses this system for service configuration
"""

import copy
import logging
import os

import yaml

all = ["Params", "load_params"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s [Params] - %(asctime)s - %(message)s")


def _deep_update_dict(source_dict, update_dict):
    """Recursively merge dictionary updates while preserving nested structure.

    This function performs intelligent dictionary merging where nested dictionaries
    are recursively updated rather than completely replaced. This enables configuration
    composition where base configurations can be extended with additional settings
    without losing existing nested data.

    The merge strategy prioritizes update_dict values when conflicts occur, but
    preserves all non-conflicting nested structure from both dictionaries.

    :param source_dict: Base dictionary to be updated in-place
    :type source_dict: dict
    :param update_dict: Dictionary containing updates and additions
    :type update_dict: dict

    .. note::
       This function modifies source_dict in-place. The source dictionary
       will contain the merged result after the function completes.

    .. warning::
       Non-dictionary values in update_dict will completely replace
       corresponding values in source_dict, even if the source value
       was a dictionary.

    Examples:
        Basic nested merging::

            >>> source = {'db': {'host': 'localhost', 'port': 5432}}
            >>> update = {'db': {'name': 'myapp'}, 'cache': {'enabled': True}}
            >>> _deep_update_dict(source, update)
            >>> print(source)
            {'db': {'host': 'localhost', 'port': 5432, 'name': 'myapp'},
             'cache': {'enabled': True}}

        Value replacement behavior::

            >>> source = {'timeout': {'connect': 5, 'read': 30}}
            >>> update = {'timeout': 10}  # Non-dict replaces dict
            >>> _deep_update_dict(source, update)
            >>> print(source['timeout'])  # 10 (not merged)

    .. seealso::
       :func:`_load_yaml` : Uses this function for configuration file merging
       :func:`load_params` : Main entry point that leverages deep merging
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in source_dict:
            # If the value is a dict and the key exists in the source,
            # perform a deep update
            _deep_update_dict(source_dict.get(key, {}), value)
        else:
            # Otherwise, set the key to the value (updating or adding)
            source_dict[key] = value


def _load_yaml(file_path, visited=[]):
    """Load YAML configuration with recursive import processing and cycle detection.

    This function implements the core configuration loading logic that supports
    hierarchical configuration files through import directives. When a YAML file
    contains an 'import' key, the specified file is loaded first, then the current
    file's configuration is merged on top using deep dictionary merging.

    The function prevents infinite recursion through circular import detection,
    maintaining a visited file list to identify and block circular dependencies.
    Import paths are resolved relative to the importing file's directory.

    :param file_path: Path to the YAML configuration file to load
    :type file_path: str
    :param visited: List of absolute file paths already visited (for cycle detection)
    :type visited: list, optional
    :raises ValueError: If a circular import is detected in the configuration chain
    :raises FileNotFoundError: If an imported file cannot be found
    :raises yaml.YAMLError: If YAML parsing fails for any file in the chain
    :return: Merged configuration dictionary from the file and all its imports
    :rtype: dict

    .. warning::
       The visited parameter uses a mutable default argument for internal
       recursion tracking. External callers should not provide this parameter.

    Examples:
        Configuration with imports::

            # database.yml
            database:
              host: ${DB_HOST}
              port: 5432

            # services.yml
            import: database.yml
            services:
              web:
                port: 8080
              api:
                port: 3000

            >>> config = _load_yaml('services.yml')
            >>> print(config['database']['host'])  # Expanded from environment
            >>> print(config['services']['web']['port'])  # 8080

        Circular import detection::

            # config_a.yml: import: config_b.yml
            # config_b.yml: import: config_a.yml
            >>> _load_yaml('config_a.yml')
            ValueError: Circular import in 'config_b.yml'

    .. seealso::
       :func:`_deep_update_dict` : Dictionary merging strategy used for imports
       :func:`load_params` : Public interface that wraps this function
       :class:`Params` : Container class for the loaded configuration data
    """

    abs_file_path = os.path.abspath(file_path)
    if abs_file_path in visited:
        raise ValueError(f"Circular import in '{file_path}'")
    newVisited = visited.copy()
    newVisited.append(abs_file_path)

    with open(file_path) as file:
        values = yaml.safe_load(file)

        if "import" in values:
            # Get the import file path
            import_file = values["import"]
            if type(import_file) is not str:
                raise ValueError("'import' must be a string")
            del values["import"]  # Remove the import key

            # Set the same directory as the file that imports the other.
            dir = os.path.dirname(file_path)
            import_file = os.path.join(dir, import_file)
            if not os.path.isfile(import_file):
                raise FileNotFoundError(f"Import file '{import_file}' not found")

            # Load the import file and update its contents with the values.
            result = _load_yaml(import_file, newVisited)
            _deep_update_dict(result, values)
            return result

        return values


def load_params(file_path):
    """Load configuration parameters from YAML file into a Params object.

    This is the main entry point for the configuration loading system. It loads
    a YAML configuration file (processing any import directives) and wraps the
    resulting data in a Params object that provides type-safe, hierarchical
    access to configuration values.

    The function handles the complete configuration loading pipeline including
    import processing, environment variable expansion, and parameter object
    creation. The resulting Params object provides dot-notation access to
    nested configuration data with built-in validation and error handling.

    :param file_path: Path to the YAML configuration file to load
    :type file_path: str
    :raises FileNotFoundError: If the configuration file cannot be found
    :raises yaml.YAMLError: If YAML parsing fails
    :raises ValueError: If circular imports are detected
    :return: Parameter container with hierarchical access to configuration data
    :rtype: Params

    Examples:
        Basic configuration loading::

            # config.yml
            database:
              host: localhost
              port: 5432
            services:
              timeout: 30
              retry_count: 3

            >>> params = load_params('config.yml')
            >>> db_host = params.database.host  # 'localhost'
            >>> timeout = params.services.timeout  # 30
            >>> invalid = params.nonexistent.key  # InvalidParam, not exception

        Environment variable expansion::

            # config.yml
            project_root: ${PROJECT_ROOT}
            data_directory: ${PROJECT_ROOT}/data

            >>> import os
            >>> os.environ['PROJECT_ROOT'] = '/home/user/project'
            >>> params = load_params('config.yml')
            >>> print(params.data_directory)  # '/home/user/project/data'

    .. seealso::
       :func:`_load_yaml` : Core YAML loading implementation
       :class:`Params` : Return type providing parameter access
       :class:`InvalidParam` : Error handling for missing configuration keys
       :mod:`deployment.container_manager` : Primary consumer of this functionality
    """
    return Params(_load_yaml(file_path), "root")


class AbstractParam:
    def __init__(self, name, parent=None):
        self._name = name
        self._parent = parent

    def get_path(self):
        if self._parent is not None:
            return f"{self._parent.get_path()}.{self._name}"
        else:
            return self._name

    def copy(self):
        return copy.deepcopy(self)

    def is_valid(self):
        raise NotImplementedError()

    def __getitem__(self, key):
        raise NotImplementedError()

    def __bool__(self):
        raise NotImplementedError()


class InvalidParam(AbstractParam):
    """Parameter object representing missing or invalid configuration data.

    This class provides graceful error handling for configuration access patterns
    where requested parameters don't exist. Instead of raising exceptions immediately,
    the system returns InvalidParam objects that maintain the access chain and
    provide meaningful error messages when finally used.

    InvalidParam objects support continued dot-notation access, allowing code to
    chain parameter lookups naturally even when intermediate parameters are missing.
    The error is only raised when the parameter is actually used (e.g., in a boolean
    context or when converted to a string).

    This approach enables defensive programming patterns where configuration access
    can be attempted optimistically, with errors handled at the point of actual use.

    :param name: Name of the missing parameter
    :type name: str
    :param parent: Parent parameter object in the access chain
    :type parent: AbstractParam, optional

    Examples:
        Graceful error handling::

            >>> params = load_params('config.yml')  # Missing 'database.timeout'
            >>> timeout = params.database.timeout  # Returns InvalidParam, no exception
            >>> if timeout:  # Now evaluates to False
            ...     print(f"Timeout: {timeout}")
            ... else:
            ...     print("Using default timeout")

        Error chain preservation::

            >>> missing = params.nonexistent.deeply.nested.value
            >>> print(missing)  # Shows path to first missing parameter
            <InvalidParam: 'root.nonexistent'>

    .. seealso::
       :class:`Params` : Valid parameter container that returns InvalidParam for missing keys
       :meth:`AbstractParam.get_path` : Path construction used in error messages
    """

    def __init__(self, name, parent=None):
        super().__init__(name, parent)
        self._name = name
        self._parent = parent

    def is_valid(self):
        """Check if this parameter is valid.

        :return: Always False for InvalidParam objects
        :rtype: bool
        """
        return False

    def __getattr__(self, key):
        """Support continued dot-notation access on invalid parameters.

        Allows chaining of parameter access even when intermediate parameters
        are missing, maintaining the error state through the access chain.

        :param key: Attribute name being accessed
        :type key: str
        :return: New InvalidParam representing the continued invalid access
        :rtype: InvalidParam

        Examples:
            Continued access on missing parameters::

                >>> invalid = params.missing.parameter
                >>> still_invalid = invalid.more.nested.access
                >>> print(still_invalid.is_valid())  # False
        """
        return InvalidParam(key, self)

    def __getitem__(self, key):
        """Raise error when bracket notation is used on invalid parameters.

        :param key: Key being accessed
        :type key: str or int
        :raises TypeError: Always raises with error message showing the invalid path

        .. note::
           Unlike dot notation (__getattr__), bracket notation immediately raises
           an error to provide clear feedback about the invalid parameter access.
        """
        raise TypeError(f"'{str(self)}'")

    def __bool__(self):
        """Evaluate InvalidParam objects as False in boolean contexts.

        :return: Always False
        :rtype: bool

        Examples:
            Boolean evaluation for error handling::

                >>> param = params.possibly.missing.value
                >>> if param:
                ...     process_value(param)
                ... else:
                ...     use_default_value()
        """
        return False

    def __repr__(self):
        """Provide clear error message showing the invalid parameter path.

        Traces back through the InvalidParam chain to find the first missing
        parameter and displays its full path for debugging.

        :return: String representation showing the invalid parameter path
        :rtype: str

        Examples:
            Error message generation::

                >>> missing = params.database.missing.timeout
                >>> print(missing)
                <InvalidParam: 'root.database.missing'>
        """
        invalid = self
        while isinstance(invalid._parent, InvalidParam):
            invalid = invalid._parent

        return f"<InvalidParam: '{invalid.get_path()}'>"


class Params(AbstractParam):
    """Primary parameter container providing hierarchical access to configuration data.

    This class wraps configuration data (dictionaries, lists, or scalar values) and
    provides type-safe, hierarchical access through dot notation and bracket notation.
    The class handles environment variable expansion, supports deep copying, and
    provides graceful error handling through InvalidParam objects.

    Params objects automatically detect the data type (dict, list, or scalar) and
    provide appropriate access methods. Nested structures are recursively wrapped
    in Params objects, creating a complete hierarchy that maintains parent-child
    relationships for path tracking and error reporting.

    Environment variable expansion is performed on string values using os.expandvars,
    allowing configuration files to reference environment variables with ${VAR} syntax.

    :param data: Configuration data to wrap (dict, list, or scalar value)
    :type data: dict or list or Any
    :param name: Name of this parameter within its parent container
    :type name: str
    :param parent: Parent parameter object, None for root parameters
    :type parent: AbstractParam, optional

    Examples:
        Dictionary access patterns::

            >>> config_data = {'database': {'host': 'localhost', 'port': 5432}}
            >>> params = Params(config_data, 'root')
            >>> host = params.database.host  # 'localhost'
            >>> port = params.database['port']  # 5432
            >>> missing = params.cache.timeout  # InvalidParam, not exception

        List access patterns::

            >>> config_data = {'servers': ['web1', 'web2', 'api1']}
            >>> params = Params(config_data, 'root')
            >>> first_server = params.servers[0]  # 'web1'
            >>> server_count = len(params.servers)  # 3

        Environment variable expansion::

            >>> config_data = {'path': '${HOME}/data', 'url': '${API_HOST}:${API_PORT}'}
            >>> params = Params(config_data, 'root')
            >>> # Environment variables are expanded when values are accessed
            >>> data_path = params.path  # '/home/user/data' (if HOME is set)

    .. seealso::
       :class:`InvalidParam` : Error handling for missing configuration keys
       :class:`AbstractParam` : Base class defining the parameter interface
       :func:`load_params` : Main entry point for creating Params from YAML files
    """

    def __init__(self, data, name, parent=None):
        super().__init__(name, parent=parent)
        self._is_dict = isinstance(data, dict)
        self._is_list = isinstance(data, list)

        if self._is_dict:
            self._data = {}
            for k, v in data.items():
                self._data[k] = self.__get_data(k, v)
        elif self._is_list:
            self._data = []
            for i, v in enumerate(data):
                self._data.append(self.__get_data(str(i), v))
        else:
            self._data = data

    def is_valid(self):
        return True

    def __bool__(self):
        return bool(self._data)

    def __repr__(self):
        return self.__get_repr(0)

    def __getattr__(self, key):
        if self._is_dict:
            if key in self._data:
                return self.__expand_vars(self._data[key])
            else:
                logging.warning(f"'{self.get_path()}' has no child '{key}'")
                return InvalidParam(key, self)
                # raise InvalidParam(f"'{self.get_path()}' has no child '{key}'")
        else:
            raise TypeError(f"{self.get_path()} is not a dict")

    def __getitem__(self, key):
        if self._is_dict or self._is_list:
            return self.__expand_vars(self._data[key])
        else:
            logging.warning(f"{self.get_path()} is not a list nor a dict")
            return InvalidParam(key, self)
            # raise InvalidParam(f"{self.get_path()} is not a list nor a dict")

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, item):
        if self._is_dict or self._is_list:
            return item in self._data
        else:
            logging.warning(f"{self.get_path()} is not a list nor a dict")
            return False

    def keys(self):
        if not self._is_dict:
            logging.warning(f"{self.get_path()} is not a dict")
            return []
        return self._data.keys()

    def values(self):
        if not self._is_dict:
            logging.warning(f"{self.get_path()} is not a dict")
            return []
        result = []
        for v in self._data.values():
            result.append(self.__expand_vars(v))
        return result

    def items(self):
        if not self._is_dict:
            logging.warning(f"{self.get_path()} is not a dict")
            return []
        return self._data.items()

    def get(self, key, default=None):
        if not self._is_dict:
            logging.warning(f"{self.get_path()} is not a dict")
            return default
        return self.__expand_vars(self._data.get(key, default))

    def __eq__(self, other):
        if isinstance(other, Params):
            return self._data == other._data
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __deepcopy__(self, memo):
        "Copies the data into its native format, without Param objects."
        return copy.deepcopy(self._data, memo)

    def __call__(self):
        return self._data

    def __get_repr(self, indent):
        pad = indent * "  "
        out = ""
        if self._is_dict:
            out += "{\n"
            for k, v in self._data.items():
                v = self.__expand_vars(v)
                v_repr = v.__get_repr(indent + 1) if isinstance(v, Params) else v
                out += f"{pad}  {k}: {v_repr}\n"
            out += f"{pad}}}"
        elif self._is_list:
            out += "["
            for i, v in enumerate(self._data):
                v = self.__expand_vars(v)
                v_repr = v.__get_repr(indent + 1) if isinstance(v, Params) else v
                comma = ", " if i < len(self._data) - 1 else ""
                if isinstance(v, Params) and v._is_dict:
                    out += f"\n{pad}  {v_repr}{comma}"
                else:
                    out += f"{v_repr}{comma}"
            out += "]"
        else:
            out += f"{self._data}"

        return out

    def __get_data(self, key, value):
        if isinstance(value, dict) or isinstance(value, list):
            return Params(value, key, self)
        else:
            return self.__expand_vars(value)

    def __expand_vars(self, value):
        if type(value) is str:
            return os.path.expandvars(value)
        return value


if __name__ == "__main__":
    import pprint
    import sys

    if len(sys.argv) > 0:
        for arg in sys.argv[1:]:
            params = load_params(arg)
            print(params)

            print(" Deep Copy:")
            cp = params.copy()
            pprint.pprint(cp)

            print("\n\n\n\n")
            value = params.A.X.Y
            print("---> ", value)
