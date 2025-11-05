#!/usr/bin/env python3
"""
Voice Agency Runner
Entry point to start the Voice Calling Agency API server.
"""

import os
import sys
import argparse
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_environment():
    """Check if required environment variables are set."""
    xai_key = os.getenv("XAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    print("Environment Check:")
    print("-" * 50)

    xai_status = "✓ Configured" if (xai_key and xai_key != "your_key") else "✗ Not configured"
    groq_status = "✓ Configured" if (groq_key and groq_key != "your_key") else "✗ Not configured"

    print(f"XAI_API_KEY:  {xai_status}")
    print(f"GROQ_API_KEY: {groq_status}")
    print("-" * 50)

    if xai_status == "✗ Not configured" and groq_status == "✗ Not configured":
        print("\nWARNING: No LLM providers configured!")
        print("Please set at least one API key in the .env file:")
        print("  - XAI_API_KEY for Grok/X.AI")
        print("  - GROQ_API_KEY for Groq")
        print("\nThe server will start but API calls will fail.\n")
        return False

    print()
    return True


def print_banner():
    """Print startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        🎤 Voice Calling Agency API Server 🎤             ║
║                                                           ║
║  Intelligent voice agents powered by LLM routing         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_info(host: str, port: int):
    """Print server information."""
    print("\nServer Information:")
    print("-" * 50)
    print(f"Host:         {host}")
    print(f"Port:         {port}")
    print(f"API Docs:     http://{host}:{port}/docs")
    print(f"Health Check: http://{host}:{port}/health")
    print(f"Root:         http://{host}:{port}/")
    print("-" * 50)

    print("\nAvailable Agent Roles:")
    print("  • sales       - Professional sales agent")
    print("  • support     - Customer support agent")
    print("  • appointment - Appointment scheduling")
    print("  • survey      - Survey agent")
    print("  • general     - General-purpose assistant")

    print("\nStarting server...\n")


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the Voice Agency API server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    print_banner()
    check_environment()
    print_info(host, port)

    # Run the server
    try:
        uvicorn.run(
            "api_server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nShutting down Voice Agency API server...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError starting server: {e}")
        sys.exit(1)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Voice Calling Agency API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with default settings (0.0.0.0:8000)
  python run_agency.py

  # Start server on custom host and port
  python run_agency.py --host 127.0.0.1 --port 8080

  # Start server with auto-reload for development
  python run_agency.py --reload

  # Check environment only
  python run_agency.py --check-env
        """
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check environment configuration and exit"
    )

    args = parser.parse_args()

    # Check environment only
    if args.check_env:
        print_banner()
        check_environment()
        sys.exit(0)

    # Run the server
    run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
