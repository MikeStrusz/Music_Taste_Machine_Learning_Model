# NMFApp Development Plan

This document outlines the plan for developing the NMFApp, a cross-platform application that connects to a user's Spotify account and provides feedback on the most recent batch of New Music Friday albums, including genre, similar artists, and a prediction for how likely they are to like it.

## Prerequisites

- **Node.js and npm**: Ensure you have Node.js and npm installed on your system.
- **React Native CLI**: Install the React Native CLI globally:
  ```bash
  npm install -g react-native-cli
  ```
- **Xcode**: Required for iOS development. Install it from the Mac App Store.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/NMFApp.git
   cd NMFApp
   ```

2. **Install Dependencies**:
   ```bash
   npm install
   ```

## Running the App

### Android

1. **Start the Metro Bundler**:
   ```bash
   npx react-native start
   ```

2. **Run the App on an Android Emulator or Device**:
   ```bash
   npx react-native run-android
   ```

### iOS

1. **Start the Metro Bundler**:
   ```bash
   npx react-native start
   ```

2. **Run the App on an iOS Simulator or Device**:
   ```bash
   npx react-native run-ios
   ```

## Spotify Integration

- **Spotify Developer Account**: Ensure you have a Spotify Developer account and have set up an application to get the necessary credentials.
- **Update API Credentials**: Update the Spotify API credentials in the app configuration.

## Making Changes

- Make your changes in the `src` directory.
- Test your changes on an emulator or device.

## Contributing

- Fork the repository.
- Create a new branch for your feature.
- Submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 