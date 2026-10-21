-- phase 1
DROP USER IF EXISTS `2nd-shift`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_2nd-shift_otel_logs` ON `telemetry`.`otel_logs`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_2nd-shift_otel_traces` ON `telemetry`.`otel_traces`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_2nd-shift_audit_ledger` ON `telemetry`.`audit_ledger`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_2nd-shift_otel_metrics_gauge` ON `telemetry`.`otel_metrics_gauge`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_2nd-shift_otel_metrics_sum` ON `telemetry`.`otel_metrics_sum`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_2nd-shift_otel_metrics_histogram` ON `telemetry`.`otel_metrics_histogram`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_2nd-shift_otel_metrics_exponential_histogram` ON `telemetry`.`otel_metrics_exponential_histogram`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_2nd-shift_otel_metrics_summary` ON `telemetry`.`otel_metrics_summary`;
-- phase 1
DROP USER IF EXISTS `cut-run`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_cut-run_otel_logs` ON `telemetry`.`otel_logs`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_cut-run_otel_traces` ON `telemetry`.`otel_traces`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_cut-run_audit_ledger` ON `telemetry`.`audit_ledger`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_cut-run_otel_metrics_gauge` ON `telemetry`.`otel_metrics_gauge`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_cut-run_otel_metrics_sum` ON `telemetry`.`otel_metrics_sum`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_cut-run_otel_metrics_histogram` ON `telemetry`.`otel_metrics_histogram`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_cut-run_otel_metrics_exponential_histogram` ON `telemetry`.`otel_metrics_exponential_histogram`;
-- phase 1
DROP ROW POLICY IF EXISTS `reader_cut-run_otel_metrics_summary` ON `telemetry`.`otel_metrics_summary`;
-- phase 1
DROP USER IF EXISTS `view__reader_x`;
-- phase 1
DROP ROW POLICY IF EXISTS `viewreader_x_otel_logs` ON `telemetry`.`otel_logs`;
-- phase 1
DROP ROW POLICY IF EXISTS `viewreader_x_otel_traces` ON `telemetry`.`otel_traces`;
-- phase 1
DROP ROW POLICY IF EXISTS `viewreader_x_audit_ledger` ON `telemetry`.`audit_ledger`;
-- phase 1
DROP ROW POLICY IF EXISTS `viewreader_x_otel_metrics_gauge` ON `telemetry`.`otel_metrics_gauge`;
-- phase 1
DROP ROW POLICY IF EXISTS `viewreader_x_otel_metrics_sum` ON `telemetry`.`otel_metrics_sum`;
-- phase 1
DROP ROW POLICY IF EXISTS `viewreader_x_otel_metrics_histogram` ON `telemetry`.`otel_metrics_histogram`;
-- phase 1
DROP ROW POLICY IF EXISTS `viewreader_x_otel_metrics_exponential_histogram` ON `telemetry`.`otel_metrics_exponential_histogram`;
-- phase 1
DROP ROW POLICY IF EXISTS `viewreader_x_otel_metrics_summary` ON `telemetry`.`otel_metrics_summary`;
-- phase 1
DROP USER IF EXISTS `history`;
-- phase 1
DROP USER IF EXISTS `ingest`;
-- phase 3
CREATE ROLE IF NOT EXISTS `readers`;
-- phase 3
CREATE ROLE IF NOT EXISTS `views`;
-- phase 3
CREATE ROLE IF NOT EXISTS `views_own`;
-- phase 3
CREATE RESOURCE IF NOT EXISTS `osprey_queries` (QUERY);
-- phase 3
CREATE OR REPLACE WORKLOAD `readers` SETTINGS max_concurrent_queries = 2;
-- phase 3
CREATE SETTINGS PROFILE IF NOT EXISTS `reader_limits`;
-- phase 3
ALTER SETTINGS PROFILE `reader_limits` SETTINGS readonly = 1 CONST, max_execution_time = 30 MAX 30 CHANGEABLE_IN_READONLY, timeout_overflow_mode = 'throw' CONST, max_result_rows = 1000 CONST, max_result_bytes = 10000000 CONST, result_overflow_mode = 'throw' CONST, max_rows_to_read = 1000000 CONST, max_bytes_to_read = 500000000 CONST, read_overflow_mode = 'throw' CONST, max_memory_usage = 200000000 CONST, max_bytes_before_external_sort = 100000000 CONST, max_bytes_before_external_group_by = 100000000 CONST, max_concurrent_queries_for_user = 2 CONST, log_queries = 1 CONST, log_queries_min_type = 'QUERY_START' CONST, log_queries_probability = 1 CONST, http_wait_end_of_query = 1 MIN 1 MAX 1 CHANGEABLE_IN_READONLY, log_comment CHANGEABLE_IN_READONLY, add_http_cors_header CHANGEABLE_IN_READONLY, enable_http_compression CHANGEABLE_IN_READONLY, framing_output_format CHANGEABLE_IN_READONLY, extremes CHANGEABLE_IN_READONLY, use_query_cache CHANGEABLE_IN_READONLY, enable_reads_from_query_cache CHANGEABLE_IN_READONLY, enable_writes_to_query_cache CHANGEABLE_IN_READONLY, query_cache_ttl CHANGEABLE_IN_READONLY, query_cache_nondeterministic_function_handling CHANGEABLE_IN_READONLY, query_cache_system_table_handling CHANGEABLE_IN_READONLY, send_logs_level CHANGEABLE_IN_READONLY, http_response_headers CHANGEABLE_IN_READONLY, workload = 'readers' CONST;
-- phase 3
CREATE QUOTA IF NOT EXISTS `reader_quota`;
-- phase 3
ALTER QUOTA `reader_quota` KEYED BY user_name FOR INTERVAL 1 MINUTE MAX queries = 60 TO `readers`;
-- phase 3
CREATE USER IF NOT EXISTS `ingest` IDENTIFIED WITH sha256_hash BY '4eb2e87756d7736d84ada0d63588dba18615261b757728296af7896d1752c552' SALT 'fb2b0ff56134f383ebe48f4a1820bd94' DEFAULT DATABASE `telemetry` SETTINGS log_queries = 0 CONST, max_partitions_per_insert_block = 100 CONST;
-- phase 3
ALTER USER `ingest` IDENTIFIED WITH sha256_hash BY '4eb2e87756d7736d84ada0d63588dba18615261b757728296af7896d1752c552' SALT 'fb2b0ff56134f383ebe48f4a1820bd94' DEFAULT DATABASE `telemetry` SETTINGS log_queries = 0 CONST, max_partitions_per_insert_block = 100 CONST;
-- phase 3
GRANT INSERT ON `telemetry`.* TO `ingest`;
-- phase 3
CREATE USER IF NOT EXISTS `archive` IDENTIFIED WITH sha256_hash BY '6377972ba60ee8faf8288ffb198d24db8367084416a09afa7677bad08b1b9390' SALT '6d2700ee0908402afd4214540c70b59e' DEFAULT DATABASE `telemetry` SETTINGS readonly = 1 CONST, max_execution_time = 600 CONST, max_memory_usage = 200000000 CONST, max_bytes_before_external_sort = 100000000 CONST, max_bytes_before_external_group_by = 100000000 CONST, output_format_parquet_row_group_size = 100000 CONST, log_queries = 1 CONST;
-- phase 3
ALTER USER `archive` IDENTIFIED WITH sha256_hash BY '6377972ba60ee8faf8288ffb198d24db8367084416a09afa7677bad08b1b9390' SALT '6d2700ee0908402afd4214540c70b59e' DEFAULT DATABASE `telemetry` SETTINGS readonly = 1 CONST, max_execution_time = 600 CONST, max_memory_usage = 200000000 CONST, max_bytes_before_external_sort = 100000000 CONST, max_bytes_before_external_group_by = 100000000 CONST, output_format_parquet_row_group_size = 100000 CONST, log_queries = 1 CONST;
-- phase 3
GRANT SELECT ON `telemetry`.* TO `archive`;
-- phase 3
GRANT SELECT ON `system`.`query_log` TO `archive`;
-- phase 3
CREATE USER IF NOT EXISTS `history` IDENTIFIED WITH sha256_hash BY 'e936c371f95b699ab4b14d70e3ab500bedec84f29090929ce09134e3b4269222' SALT 'ef57927baad31ce2030e9ca681034898' DEFAULT DATABASE `telemetry` SETTINGS PROFILE 'history_limits';
-- phase 3
ALTER USER `history` IDENTIFIED WITH sha256_hash BY 'e936c371f95b699ab4b14d70e3ab500bedec84f29090929ce09134e3b4269222' SALT 'ef57927baad31ce2030e9ca681034898' DEFAULT DATABASE `telemetry` SETTINGS PROFILE 'history_limits';
-- phase 3
CREATE USER IF NOT EXISTS `jean-luc` IDENTIFIED WITH sha256_hash BY '555231c4e35db379f69f4fcf44ea9a436da295f9cca371224f24cf6b66f10b3c' SALT '2a18d71e3eac8e6f6de163012737422b' DEFAULT DATABASE `telemetry` SETTINGS PROFILE 'reader_limits';
-- phase 3
ALTER USER `jean-luc` IDENTIFIED WITH sha256_hash BY '555231c4e35db379f69f4fcf44ea9a436da295f9cca371224f24cf6b66f10b3c' SALT '2a18d71e3eac8e6f6de163012737422b' DEFAULT DATABASE `telemetry` SETTINGS PROFILE 'reader_limits';
-- phase 3
GRANT `readers` TO `jean-luc`;
-- phase 3
CREATE ROW POLICY OR REPLACE `reader_jean-luc_otel_logs` ON `telemetry`.`otel_logs` AS RESTRICTIVE USING `Timestamp` >= now() - INTERVAL 14 DAY TO `jean-luc`;
-- phase 3
GRANT SELECT ON `telemetry`.`otel_logs` TO `jean-luc`;
-- phase 3
CREATE ROW POLICY OR REPLACE `reader_jean-luc_otel_traces` ON `telemetry`.`otel_traces` AS RESTRICTIVE USING `Timestamp` >= now() - INTERVAL 14 DAY TO `jean-luc`;
-- phase 3
GRANT SELECT ON `telemetry`.`otel_traces` TO `jean-luc`;
-- phase 3
CREATE ROW POLICY OR REPLACE `reader_jean-luc_audit_ledger` ON `telemetry`.`audit_ledger` AS RESTRICTIVE USING `Timestamp` >= now() - INTERVAL 14 DAY TO `jean-luc`;
-- phase 3
GRANT SELECT ON `telemetry`.`audit_ledger` TO `jean-luc`;
