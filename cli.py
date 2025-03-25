#!/usr/bin/env python3
"""
Advanced Terminal User Interface

High-performance interactive TUI with real-time streaming, keyboard shortcuts,
and multi-modal features for an exceptional local LLM experience.
"""

import os
import sys
import asyncio
import argparse
import logging
import json
import threading
import signal
import time
from enum import Enum
from typing import Optional, List, Dict, Any, Union, Callable, AsyncGenerator
from pathlib import Path
from datetime import datetime
import re

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("tui")

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent))

# Import local modules
try:
    from models import model_manager
    from rag import get_context
except ImportError as e:
    logger.error(f"Error importing modules: {str(e)}")
    print(f"Failed to import required modules: {str(e)}")
    sys.exit(1)

# Import UI frameworks with graceful fallbacks
try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.widgets import Header, Footer, Button, Static, Input, Label, Markdown, Rule
    from textual.reactive import reactive
    from textual.binding import Binding
    from textual import events, work
    from textual.screen import Screen
    from textual.widget import Widget
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.prompt import Prompt
        from rich.table import Table
        from rich.progress import Progress, SpinnerColumn, TextColumn
        from rich.live import Live
        from rich import box
        RICH_AVAILABLE = True
    except ImportError:
        RICH_AVAILABLE = False
        print("For a better experience, install textual: pip install textual")

# Global console for basic rich formatting fallback
console = Console() if RICH_AVAILABLE else None

class ChatMode(Enum):
    """Different chat modes available to the user"""
    STANDARD = "standard"
    RAG = "rag"
    CREATIVE = "creative"
    CONCISE = "concise"

class ChatHistory:
    """Enhanced chat history manager with storage and analysis capabilities"""
    
    def __init__(self, persona: str):
        """Initialize with specific persona"""
        self.persona = persona
        self.history = []
        self.filename = None
        self.storage_path = Path(os.getenv("CHAT_HISTORY_PATH", "chat_logs"))
        
        # Ensure storage path exists
        self.storage_path.mkdir(exist_ok=True, parents=True)
    
    def add_exchange(self, user_input: str, response: str, mode: ChatMode = ChatMode.STANDARD):
        """Add a user-bot exchange to history with metadata"""
        timestamp = datetime.now().isoformat()
        exchange = {
            "timestamp": timestamp,
            "user": user_input,
            "bot": response,
            "mode": mode.value,
            "persona": self.persona
        }
        self.history.append(exchange)
    
    def save_to_file(self, filename: Optional[str] = None, format: str = "markdown") -> Path:
        """Save chat history to file in specified format"""
        if not filename and not self.filename:
            # Generate filename based on persona and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if format == "json":
                filename = f"chat_{self.persona}_{timestamp}.json"
            else:
                filename = f"chat_{self.persona}_{timestamp}.md"
            self.filename = filename
        
        filepath = self.storage_path / (filename or self.filename)
        
        # Ensure the directory exists
        filepath.parent.mkdir(exist_ok=True)
        
        # Write history to file in requested format
        if format == "json":
            self._save_as_json(filepath)
        else:
            self._save_as_markdown(filepath)
        
        return filepath
    
    def _save_as_json(self, filepath: Path):
        """Save chat history in JSON format"""
        with open(filepath, "w") as f:
            json.dump({
                "persona": self.persona,
                "display_name": model_manager.get_display_name(self.persona),
                "timestamp": datetime.now().isoformat(),
                "history": self.history
            }, f, indent=2)
    
    def _save_as_markdown(self, filepath: Path):
        """Save chat history in markdown format"""
        with open(filepath, "w") as f:
            f.write(f"# Chat with {model_manager.get_display_name(self.persona)}\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for exchange in self.history:
                f.write(f"## User ({exchange['timestamp']})\n\n")
                f.write(f"{exchange['user']}\n\n")
                
                # Add mode if not standard
                if exchange.get('mode') != ChatMode.STANDARD.value:
                    f.write(f"*Mode: {exchange.get('mode')}*\n\n")
                    
                f.write(f"## {model_manager.get_display_name(self.persona)}\n\n")
                f.write(f"{exchange['bot']}\n\n")
                f.write("---\n\n")
    
    def get_context_window(self, window_size: int = 5) -> str:
        """Get recent conversation history for context window"""
        if not self.history:
            return ""
            
        context_exchanges = self.history[-window_size:]
        context_text = []
        
        for exchange in context_exchanges:
            context_text.append(f"User: {exchange['user']}")
            context_text.append(f"{model_manager.get_display_name(self.persona)}: {exchange['bot']}")
        
        return "\n".join(context_text)
    
    def clear(self):
        """Clear the chat history"""
        self.history = []
        
    def get_last_exchange(self) -> Optional[Dict]:
        """Get the last exchange in history"""
        if not self.history:
            return None
        return self.history[-1]

# Textual TUI implementation (preferred)
if TEXTUAL_AVAILABLE:
    class ChatMessage(Static):
        """A widget to display a chat message"""
        
        def __init__(self, message: str, is_user: bool = False):
            super().__init__()
            self.message = message
            self.is_user = is_user
            
        def compose(self) -> ComposeResult:
            # Style based on message source
            style = "white on dark_blue" if self.is_user else "white on dark_green"
            sender = "You" if self.is_user else model_manager.get_display_name(self.app.persona)
            
            # Create header with sender
            yield Static(f"[b]{sender}[/b]", classes="message-header")
            
            # Format and display message
            formatted = Markdown(self.message)
            yield Static(formatted, classes="message-content")
    
    class ChatModeButton(Button):
        """A specialized button for selecting chat modes"""
        def __init__(self, mode: ChatMode):
            super().__init__(mode.value.capitalize())
            self.mode = mode
    
    class ChatWindow(ScrollableContainer):
        """The main chat message display area"""
        
        def __init__(self):
            super().__init__(id="chat-window")
            
        def add_message(self, message: str, is_user: bool = False):
            """Add a message to the chat window"""
            self.mount(ChatMessage(message, is_user))
            self.scroll_end(animate=True)
    
    class ChatInputBar(Horizontal):
        """Input bar with send button and mode selector"""
        
        def compose(self) -> ComposeResult:
            yield Input(placeholder="Type your message here...", id="chat-input")
            yield Button("Send", id="send-button", variant="primary")
    
    class ChatModeSelector(Horizontal):
        """Mode selector strip"""
        
        def compose(self) -> ComposeResult:
            for mode in ChatMode:
                yield ChatModeButton(mode)
    
    class StatusBar(Horizontal):
        """Status bar showing current state and persona"""
        
        def __init__(self, persona: str):
            super().__init__(id="status-bar")
            self.persona = persona
            
        def compose(self) -> ComposeResult:
            yield Static(f"Persona: [b]{model_manager.get_display_name(self.persona)}[/b]", id="persona-indicator")
            yield Static("Status: Ready", id="status-indicator")
            yield Static("Mode: Standard", id="mode-indicator")
    
    class ChatApp(App):
        """Advanced TUI chat application"""
        
        TITLE = "AI Political Chat"
        CSS = """
        #chat-window {
            height: 1fr;
            border: solid green;
            padding: 1;
            overflow-y: auto;
        }
        
        #status-bar {
            height: 1;
            dock: bottom;
            background: $primary-background;
            color: $text;
            padding: 0 1;
        }
        
        .message-header {
            padding: 0 1;
            background: $primary-background-lighten-2;
            color: $text;
            margin-bottom: 1;
        }
        
        .message-content {
            padding: 0 1 1 1;
            margin-bottom: 1;
        }
        
        Button {
            margin: 0 1 0 0;
        }
        """
        
        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("ctrl+s", "save", "Save Chat"),
            Binding("ctrl+c", "copy", "Copy Text"),
            Binding("ctrl+r", "toggle_rag", "Toggle RAG"),
            Binding("ctrl+l", "clear", "Clear Chat"),
            Binding("ctrl+h", "help", "Help"),
            Binding("enter", "send", "Send")
        ]
        
        def __init__(self, args):
            """Initialize with command line arguments"""
            super().__init__()
            self.args = args
            self.persona = args.persona
            self.current_mode = ChatMode.RAG if args.rag else ChatMode.STANDARD
            self.chat_history = ChatHistory(self.persona)
            self.processing = False
            
        def compose(self) -> ComposeResult:
            """Build the UI layout"""
            yield Header()
            yield ChatWindow()
            yield Rule()
            yield ChatModeSelector()
            yield ChatInputBar()
            yield StatusBar(self.persona)
            yield Footer()
        
        def on_mount(self) -> None:
            """Initialize the application state"""
            self.update_status("Ready")
            self.update_mode_indicator()
            
            # Show initial greeting
            chat_window = self.query_one("#chat-window", ChatWindow)
            greeting = f"Hello! I'm AI {model_manager.get_display_name(self.persona)}. How can I help you today?"
            chat_window.add_message(greeting)
        
        def update_status(self, status: str) -> None:
            """Update the status indicator"""
            self.query_one("#status-indicator", Static).update(f"Status: {status}")
        
        def update_mode_indicator(self) -> None:
            """Update the mode indicator"""
            self.query_one("#mode-indicator", Static).update(f"Mode: {self.current_mode.value.capitalize()}")
        
        def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle button presses"""
            button_id = event.button.id
            
            if button_id == "send-button":
                self.send_message()
            elif isinstance(event.button, ChatModeButton):
                self.current_mode = event.button.mode
                self.update_mode_indicator()
        
        def on_key(self, event: events.Key) -> None:
            """Handle key press events"""
            if event.key == "enter" and not self.processing:
                self.send_message()
        
        def action_send(self) -> None:
            """Send message action"""
            if not self.processing:
                self.send_message()
        
        def action_toggle_rag(self) -> None:
            """Toggle RAG mode"""
            if self.current_mode == ChatMode.RAG:
                self.current_mode = ChatMode.STANDARD
            else:
                self.current_mode = ChatMode.RAG
            self.update_mode_indicator()
        
        def action_clear(self) -> None:
            """Clear chat history"""
            self.chat_history.clear()
            chat_window = self.query_one("#chat-window", ChatWindow)
            chat_window.remove_children()
            
            # Show new greeting
            greeting = f"Chat cleared. How can I help you today?"
            chat_window.add_message(greeting)
        
        def action_save(self) -> None:
            """Save chat history"""
            if not self.chat_history.history:
                self.notify("No chat history to save")
                return
                
            filepath = self.chat_history.save_to_file()
            self.notify(f"Chat saved to: {filepath}")
        
        def action_help(self) -> None:
            """Show help information"""
            help_text = """
            # Chat Commands
            
            - **Ctrl+S**: Save chat history
            - **Ctrl+L**: Clear chat
            - **Ctrl+R**: Toggle factual mode
            - **Ctrl+H**: Show this help
            - **Ctrl+C**: Copy selected text
            - **Q**: Quit application
            
            ## Chat Modes
            
            - **Standard**: Normal conversation
            - **RAG**: Enhanced with factual information
            - **Creative**: More imaginative responses
            - **Concise**: Brief, to-the-point answers
            """
            
            chat_window = self.query_one("#chat-window", ChatWindow)
            chat_window.add_message(help_text)
        
        def send_message(self) -> None:
            """Send a message and get a response"""
            # Get input text
            input_widget = self.query_one("#chat-input", Input)
            message = input_widget.value.strip()
            
            if not message:
                return
                
            # Clear input
            input_widget.value = ""
            
            # Display user message
            chat_window = self.query_one("#chat-window", ChatWindow)
            chat_window.add_message(message, is_user=True)
            
            # Start processing
            self.processing = True
            self.update_status("Thinking...")
            
            # Process in background
            self.process_message(message)
        
        @work(thread=True)
        async def process_message(self, message: str) -> None:
            """Process user message and generate response"""
            try:
                # Check for RAG mode to determine if we need context
                use_rag = self.current_mode == ChatMode.RAG
                
                # Get context if in RAG mode
                context = await get_context(message, self.persona) if use_rag else None
                
                # Adjust generation parameters based on mode
                max_length = self.args.max_length
                temperature = self.args.temperature
                
                if self.current_mode == ChatMode.CREATIVE:
                    temperature = min(0.9, temperature * 1.3)  # Increase temperature
                elif self.current_mode == ChatMode.CONCISE:
                    max_length = max_length // 2  # Shorter responses
                    temperature = max(0.3, temperature * 0.8)  # Lower temperature
                
                # Get response with streaming if enabled
                if self.args.stream:
                    self.update_status("Generating response...")
                    
                    # Create a placeholder for streaming text
                    chat_window = self.query_one("#chat-window", ChatWindow)
                    placeholder = ChatMessage("", False)
                    chat_window.mount(placeholder)
                    
                    # Stream response
                    response_text = ""
                    async for chunk in model_manager.generate_response(
                        self.persona, message, context, 
                        max_length, temperature, streaming=True
                    ):
                        response_text += chunk
                        placeholder.update(Markdown(response_text))
                        chat_window.scroll_end(animate=False)
                        
                else:
                    # Generate full response
                    self.update_status("Generating response...")
                    response_text = await model_manager.generate_response(
                        self.persona, message, context,
                        max_length, temperature, streaming=False
                    )
                    
                    # Display response
                    chat_window = self.query_one("#chat-window", ChatWindow)
                    chat_window.add_message(response_text)
                
                # Add to history
                self.chat_history.add_exchange(message, response_text, self.current_mode)
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                chat_window = self.query_one("#chat-window", ChatWindow)
                chat_window.add_message(f"Error: {str(e)}")
            
            finally:
                # Finish processing
                self.processing = False
                self.update_status("Ready")
    
    def run_tui(args):
        """Run the Textual TUI"""
        app = ChatApp(args)
        app.run()

# Rich-based CLI fallback
def print_header(persona: str, use_rag: bool):
    """Print application header with styling"""
    display_name = model_manager.get_display_name(persona)
    
    if RICH_AVAILABLE:
        console.print()
        console.print(Panel.fit(
            f"[bold blue]🇺🇸 Chat with {display_name} 🇺🇸[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        ))
        
        # Print configuration
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Persona", display_name)
        table.add_row("Factual Enhancement", "Enabled" if use_rag else "Disabled")
        
        console.print(table)
        
        # Print usage instructions
        console.print(Panel(
            "[bold]Commands:[/bold]\n"
            "- [cyan]/exit[/cyan] or [cyan]/quit[/cyan]: Exit the chat\n"
            "- [cyan]/clear[/cyan]: Clear the chat history\n"
            "- [cyan]/save[/cyan]: Save the chat history\n"
            "- [cyan]/toggle[/cyan]: Toggle factual enhancement\n"
            "- [cyan]/mode[/cyan] <mode>: Change mode (standard, rag, creative, concise)\n"
            "- [cyan]/help[/cyan]: Show this help",
            title="Usage",
            border_style="dim",
            expand=False
        ))
    else:
        print(f"\n🇺🇸 Chat with {display_name} 🇺🇸")
        print("=" * 30)
        print(f"Persona: {display_name}")
        print(f"Factual Enhancement: {'Enabled' if use_rag else 'Disabled'}")
        print("\nCommands:")
        print("- /exit or /quit: Exit the chat")
        print("- /clear: Clear the chat history")
        print("- /save: Save the chat history")
        print("- /toggle: Toggle factual enhancement")
        print("- /mode <mode>: Change mode (standard, rag, creative, concise)")
        print("- /help: Show this help")
    
    print()

def print_help():
    """Print help information"""
    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold]Available Commands:[/bold]\n"
            "- [cyan]/exit[/cyan] or [cyan]/quit[/cyan]: Exit the chat\n"
            "- [cyan]/clear[/cyan]: Clear the chat history\n"
            "- [cyan]/save[/cyan]: Save the chat history\n"
            "- [cyan]/toggle[/cyan]: Toggle factual enhancement\n"
            "- [cyan]/mode[/cyan] <mode>: Change mode (standard, rag, creative, concise)\n"
            "- [cyan]/help[/cyan]: Show this help\n\n"
            "[bold]Available Modes:[/bold]\n"
            "- [green]standard[/green]: Normal conversation\n"
            "- [green]rag[/green]: Enhanced with factual information\n"
            "- [green]creative[/green]: More imaginative responses\n"
            "- [green]concise[/green]: Brief, to-the-point answers\n\n"
            "[bold]Usage Tips:[/bold]\n"
            "- Ask concise, clear questions for better responses\n"
            "- Try questions about policies, positions, and achievements\n"
            "- Enable factual enhancement for more accurate information",
            title="Help",
            border_style="green",
            expand=False
        ))
    else:
        print("\nAvailable Commands:")
        print("- /exit or /quit: Exit the chat")
        print("- /clear: Clear the chat history")
        print("- /save: Save the chat history")
        print("- /toggle: Toggle factual enhancement")
        print("- /mode <mode>: Change mode (standard, rag, creative, concise)")
        print("- /help: Show this help")
        
        print("\nAvailable Modes:")
        print("- standard: Normal conversation")
        print("- rag: Enhanced with factual information")
        print("- creative: More imaginative responses")
        print("- concise: Brief, to-the-point answers")
        
        print("\nUsage Tips:")
        print("- Ask concise, clear questions for better responses")
        print("- Try questions about policies, positions, and achievements")
        print("- Enable factual enhancement for more accurate information")
    
    print()

def print_user_input(text: str):
    """Print user input with styling"""
    if RICH_AVAILABLE:
        console.print(f"[bold green]You[/bold green] > {text}")
    else:
        print(f"You > {text}")

def print_bot_response(text: str, persona: str):
    """Print bot response with styling"""
    display_name = model_manager.get_display_name(persona)
    
    if RICH_AVAILABLE:
        console.print(f"[bold blue]{display_name}[/bold blue] > {text}")
    else:
        print(f"{display_name} > {text}")

async def chat_loop(args):
    """Main chat loop with async support"""
    persona = args.persona
    use_rag = args.rag
    max_length = args.max_length
    temperature = args.temperature
    streaming = args.stream
    chat_history = ChatHistory(persona)
    current_mode = ChatMode.RAG if use_rag else ChatMode.STANDARD
    
    # Print header
    print_header(persona, use_rag)
    
    # Print welcome message
    print_bot_response(f"Hello! I'm AI {model_manager.get_display_name(persona)}. How can I help you today?", persona)
    
    # Setup signal handler for graceful exit
    def signal_handler(sig, frame):
        print("\nInterrupted by user. Saving chat history and exiting...")
        if chat_history.history:
            filepath = chat_history.save_to_file()
            print(f"Chat history saved to: {filepath}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    while True:
        try:
            # Get user input
            if RICH_AVAILABLE:
                user_input = Prompt.ask(f"[bold green]You[/bold green] > ")
            else:
                user_input = input("You > ")
                
            user_input = user_input.strip()
            
            # Check for commands
            if user_input.lower() in ['/exit', '/quit']:
                if chat_history.history:
                    save = input("Save chat history before exiting? (y/n): ").lower() == 'y'
                    if save:
                        filepath = chat_history.save_to_file()
                        print(f"Chat history saved to: {filepath}")
                break
                
            if user_input.lower() == '/help':
                print_help()
                continue
                
            if user_input.lower() == '/clear':
                chat_history.clear()
                print("Chat history cleared.")
                continue
                
            if user_input.lower() == '/save':
                if not chat_history.history:
                    print("No chat history to save.")
                    continue
                    
                format_choice = input("Save as markdown or json? (md/json): ").lower()
                format = "json" if format_choice == "json" else "markdown"
                
                filepath = chat_history.save_to_file(format=format)
                print(f"Chat history saved to: {filepath}")
                continue
                
            if user_input.lower() == '/toggle':
                use_rag = not use_rag
                current_mode = ChatMode.RAG if use_rag else ChatMode.STANDARD
                status = "enabled" if use_rag else "disabled"
                print(f"Factual enhancement {status}.")
                continue
                
            if user_input.lower().startswith('/mode '):
                mode_name = user_input.lower().split(' ', 1)[1].strip()
                try:
                    current_mode = ChatMode(mode_name)
                    use_rag = (current_mode == ChatMode.RAG)
                    print(f"Mode changed to: {current_mode.value}")
                except ValueError:
                    print(f"Unknown mode: {mode_name}")
                    print(f"Available modes: {', '.join(m.value for m in ChatMode)}")
                continue
                
            if not user_input:
                continue
            
            # Echo user input
            print_user_input(user_input)
            
            # Adjust parameters based on mode
            mode_max_length = max_length
            mode_temperature = temperature
            
            if current_mode == ChatMode.CREATIVE:
                mode_temperature = min(0.9, temperature * 1.3)
            elif current_mode == ChatMode.CONCISE:
                mode_max_length = max_length // 2
                mode_temperature = max(0.3, temperature * 0.8)
            
            # Indicate processing
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Thinking..."),
                    transient=True
                ) as progress:
                    progress.add_task("thinking", total=None)
                    
                    # Get context if RAG is enabled
                    context = await get_context(user_input, persona) if use_rag else None
                    
                    if streaming:
                        # Start generating response with streaming
                        response_text = ""
                        with console.status(f"[bold blue]{model_manager.get_display_name(persona)} is typing...[/bold blue]"):
                            # Print persona name
                            console.print(f"[bold blue]{model_manager.get_display_name(persona)}[/bold blue] > ", end="")
                            
                            # Stream response
                            async for chunk in model_manager.generate_response(
                                persona, user_input, context, 
                                mode_max_length, mode_temperature, streaming=True
                            ):
                                console.print(chunk, end="")
                                response_text += chunk
                            
                            # Add newline at the end
                            console.print()
                    else:
                        # Generate complete response
                        response_text = await model_manager.generate_response(
                            persona, user_input, context,
                            mode_max_length, mode_temperature, streaming=False
                        )
                        
                        # Print response
                        print_bot_response(response_text, persona)
            else:
                # Simpler version without Rich
                print("Thinking...")
                
                # Get context if RAG is enabled
                context = await get_context(user_input, persona) if use_rag else None
                
                # Generate complete response
                response_text = await model_manager.generate_response(
                    persona, user_input, context,
                    mode_max_length, mode_temperature, streaming=False
                )
                
                # Print response
                print_bot_response(response_text, persona)
            
            # Add to history
            chat_history.add_exchange(user_input, response_text, current_mode)
            
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            print(f"Error: {str(e)}")
    
    # Cleanup
    model_manager.clear_cache(persona)
    print("Goodbye!")

def main():
    """Main entry point for the CLI application"""
    parser = argparse.ArgumentParser(
        description="Advanced chat interface for AI political personas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments
    parser.add_argument(
        "--persona", 
        choices=model_manager.get_available_personas(),
        default="trump",
        help="Which political persona to chat with"
    )
    
    parser.add_argument(
        "--rag", 
        action="store_true",
        help="Enable factual enhancement using RAG"
    )
    
    parser.add_argument(
        "--max-length", 
        type=int,
        default=512,
        help="Maximum length of generated responses"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float,
        default=0.7,
        help="Temperature for response generation (0.0-1.0)"
    )
    
    parser.add_argument(
        "--stream", 
        action="store_true",
        help="Enable response streaming for real-time output"
    )
    
    parser.add_argument(
        "--tui", 
        action="store_true",
        help="Use advanced terminal user interface (requires textual)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Use TUI if specified and available
    if args.tui and TEXTUAL_AVAILABLE:
        run_tui(args)
    else:
        # Fall back to standard chat loop
        asyncio.run(chat_loop(args))

if __name__ == "__main__":
    main() 