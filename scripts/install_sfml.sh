#!/bin/bash
rm -r ~/local/sfml/
# Set installation directory
INSTALL_DIR="$HOME/local/sfml"

# Check if SFML is already installed
if [ -d "$INSTALL_DIR" ]; then
    echo "SFML already installed in $INSTALL_DIR"
    exit 0
fi

# Create directories
mkdir -p "$HOME/Downloads"
mkdir -p "$INSTALL_DIR"

# Download SFML source using tar.gz
cd "$HOME/Downloads"
#wget https://www.sfml-dev.org/files/SFML-2.5.1-sources.tar.gz
wget https://www.sfml-dev.org/files/SFML-2.5.1-linux-gcc-64-bit.tar.gz
tar xzf SFML-2.5.1-linux-gcc-64-bit.tar.gz

# Build and install SFML
cd SFML-2.5.1
cmake -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" .
make -j4
make install

# Add library path to bashrc if not already present
if ! grep -q "export LD_LIBRARY_PATH.*$INSTALL_DIR/lib" "$HOME/.bashrc"; then
    echo "export LD_LIBRARY_PATH=$INSTALL_DIR/lib:\$LD_LIBRARY_PATH" >> "$HOME/.bashrc"
fi

# Clean up
# cd ..
# rm -rf SFML-2.5.1*

echo "SFML installed successfully in $INSTALL_DIR"
echo "Please run: source ~/.bashrc"