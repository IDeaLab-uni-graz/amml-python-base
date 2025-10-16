{}: let
  pkgs = import <nixpkgs> {};
  unstablePkgs = import <nixos-unstable> {};

  lib = pkgs.lib;
  stdenv = pkgs.stdenv;

  config = import <config> {};
  configPath = builtins.getEnv "NIXOS_CONFIGURATION_DIR" + "/.";
  currentDir = ./.;
in
  pkgs.mkShell {
    buildInputs = [
      pkgs.act
	  pkgs.gh
    ];

    shellHook = ''
      echo "Welcome to the AMML Docker base image shell"
      export PS1="\\[\\033[1;36m\\][\\u@AMML Python Base(\\h):\\w]$\\[\\033[0m\\] "
      export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
    '';
  }
