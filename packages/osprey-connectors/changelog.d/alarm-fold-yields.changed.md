The Kafka alarm feed folds a large batch of messages in chunks and gives the event loop a turn between them, so reading a long topic history no longer holds the process's other tasks.
