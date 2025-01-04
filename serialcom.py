import serial
import time

# Configure the serial port
ser = serial.Serial(
    port='/dev/ttyUSB0',   # Change to your specific serial port
    baudrate=115200,       # Baudrate
    timeout=1              # Timeout for reading
)

def write_data(data):
    """Write data to the serial port."""
    if ser.is_open:
        ser.write(data.encode('utf-8'))
        print(f"Sent: {data}")
    else:
        print("Serial port is not open!")

def read_data():
    """Read data from the serial port."""
    if ser.is_open:
        data = ser.readline().decode('utf-8').rstrip()
        if data:
            print(f"Received: {data}")
        return data
    else:
        print("Serial port is not open!")
        return None

if __name__ == "__main__":
    try:
        # Example usage: writing and reading data
        while True:
            write_data("Hello, Serial Port!")  # Send data
            time.sleep(0.001)  # Wait for a second
            read_data()    # Read incoming data

    except KeyboardInterrupt:
        print("Exiting program.")

    finally:
        ser.close()  # Close the serial port when done
