"""Version constants that affect compiled behavior."""

# Increment when prompt rendering changes in a way that can change model behavior.
# 3: untrusted-content labels, protocol-marker neutralization, and size caps (T-0029).
RENDERER_VERSION = "3"

# Increment when artifact parsing changes in a way that can change the stored artifact.
PARSER_VERSION = "2"
