
STOP_POLICY := INSTRUCTION_INTERVAL

I_START1 := 39005308705
I_END1   := 101363602583

I_START2 := 23819757615
I_END2   := 79972839425

# Empty: benchmark 2 uses its native memory-page configuration.
# Its checkpoint is stored under the "native" checkpoint layout.

# Temporary 1GB co-runner validation
SMT_CORUNNER_LAYOUT := layout1gb
DEFAULT_NUM_OF_REPEATS := 1
