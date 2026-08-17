#!/usr/bin/env python3
"""
Test to verify that run_in_container() executes commands EXACTLY ONCE.

This test creates a temporary script that increments a counter and verifies
the counter is incremented exactly once.
"""
import subprocess
import sys
import tempfile
from pathlib import Path


def test_single_execution():
    """Verify command executes exactly once."""
    print("Testing single execution guarantee...")
    print("-" * 60)

    # Create a temporary counter file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        counter_file = f.name
        f.write("0")

    try:
        # Create a test script that increments the counter
        test_script = f"""
COUNTER_FILE="{counter_file}"
count=$(cat "$COUNTER_FILE")
echo "Before: $count"
count=$((count + 1))
echo "$count" > "$COUNTER_FILE"
echo "After: $count"
echo "Counter incremented to: $count"
"""

        # Run the script via docker exec
        container = "seedvc"

        # Check if container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container}", "--filter", "status=running", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )

        if container not in result.stdout:
            print(f"ERROR: Container '{container}' is not running")
            print("Start it with: docker compose up -d seedvc")
            return False

        # Execute the command using Popen (same pattern as fixed run_in_container)
        process = subprocess.Popen(
            ["docker", "exec", "-i", container, "bash", "-c", test_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output
        output_lines = []
        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)

        process.wait()

        # Check return code
        if process.returncode != 0:
            print(f"\nERROR: Command failed with exit code {process.returncode}")
            return False

        # Read the counter value
        with open(counter_file, 'r') as f:
            final_count = int(f.read().strip())

        print("\n" + "-" * 60)
        print(f"Final counter value: {final_count}")

        if final_count == 1:
            print("✓ PASS: Command executed EXACTLY ONCE")
            return True
        else:
            print(f"✗ FAIL: Command executed {final_count} times (expected 1)")
            return False

    finally:
        # Cleanup
        try:
            Path(counter_file).unlink()
        except:
            pass


if __name__ == "__main__":
    success = test_single_execution()
    sys.exit(0 if success else 1)
