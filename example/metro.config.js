const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');
const path = require('path');

const libraryDir = path.resolve(__dirname, '..');

const config = {
  watchFolders: [libraryDir],
  resolver: {
    unstable_enableSymlinks: true,
    nodeModulesPaths: [
      path.join(libraryDir, 'node_modules'),
      path.resolve(__dirname, 'node_modules'),
    ],
  },
};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);