"""
Local vision provider package.

This package contains provider implementations that execute
vision inference through locally hosted runtimes.

The package intentionally keeps runtime-specific integrations,
such as Ollama, isolated from the application-level VisionService.
"""