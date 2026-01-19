import time
import json
import os
import sys
from datetime import datetime

try:
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Console
    from rich import box
    from rich.text import Text
    from rich.progress import ProgressBar
except ImportError:
    print("Please install 'rich' library: pip install rich")
    sys.exit(1)

LOG_FILE = "logs/brain_state.jsonl"

def make_layout():
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1)
    )
    layout["left"].split(
        Layout(name="chemicals", ratio=1),
        Layout(name="genes", ratio=1),
        Layout(name="vm", ratio=1)
    )
    layout["right"].split(
        Layout(name="task", ratio=1),
        Layout(name="events", ratio=2)
    )
    return layout

class Dashboard:
    def __init__(self):
        self.events = []
        self.last_state = None
        
    def update(self):
        # Read new lines from log
        if not os.path.exists(LOG_FILE):
            return
            
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            if not lines: return
            
            # Process all lines to get latest state and accumulate events
            for line in lines:
                try:
                    data = json.loads(line)
                    if data.get('type') == 'EVENT':
                        event_str = f"[{datetime.fromtimestamp(data['timestamp']).strftime('%H:%M:%S')}] {data['event_type']}: {data['message']}"
                        if event_str not in self.events:
                            self.events.append(event_str)
                            if len(self.events) > 20: self.events.pop(0)
                    else:
                        self.last_state = data
                except:
                    pass
                    
    def render_header(self):
        if not self.last_state: return Panel("Waiting for Brain...")
        age = self.last_state.get('age', 0)
        return Panel(f"NEURAL COMPUTER V3.0 | Age: {age:.2f} | Status: ONLINE", style="bold white on blue")

    def render_chemicals(self):
        if not self.last_state: return Panel("No Data")
        chem = self.last_state.get('chemicals', {})
        table = Table(box=box.SIMPLE)
        table.add_column("Chemical", style="cyan")
        table.add_column("Level", style="green")
        
        for k, v in chem.items():
            bar = "█" * int(v * 10)
            table.add_row(k.capitalize(), f"{v:.2f} {bar}")
            
        return Panel(table, title="Neurochemistry")

    def render_genes(self):
        if not self.last_state: return Panel("No Data")
        genes = self.last_state.get('genes', {})
        table = Table(box=box.SIMPLE)
        table.add_column("Gene", style="magenta")
        table.add_column("Expr", style="yellow")
        
        for k, v in genes.items():
            table.add_row(k.upper(), f"{v:.2f}")
            
        return Panel(table, title="Gene Expression")

    def render_vm(self):
        if not self.last_state: return Panel("No Data")
        vm = self.last_state.get('vm', {})
        active = vm.get('active_npus', 0)
        load = vm.get('load', 0)
        
        text = Text()
        text.append(f"Active NPUs: {active}\n", style="bold green")
        text.append(f"System Load: {load:.2f}s\n", style="bold red" if load > 0.1 else "green")
        
        # Visual Grid of NPUs
        grid = ""
        for i in range(64):
            if i < active:
                grid += "[green]■[/] "
            else:
                grid += "[grey]□[/] "
            if (i+1) % 8 == 0: grid += "\n"
            
        text.append("\nNPU Grid:\n")
        text.append(grid)
        
        return Panel(text, title="Neural VM Status")

    def render_task(self):
        if not self.last_state: return Panel("No Data")
        task = self.last_state.get('task', {})
        
        text = Text()
        text.append(f"Current Task: {task.get('desc', 'Idle')}\n", style="bold white")
        text.append(f"Difficulty: {task.get('difficulty', 1)}\n", style="cyan")
        text.append(f"Loss: {task.get('loss', 0):.4f}\n", style="red")
        text.append(f"Confidence: {task.get('confidence', 0.0):.2f}\n", style="yellow")
        
        return Panel(text, title="Cognitive Task")

    def render_events(self):
        text = Text()
        for e in reversed(self.events):
            text.append(e + "\n")
        return Panel(text, title="System Events")

def run_dashboard():
    dashboard = Dashboard()
    layout = make_layout()
    
    with Live(layout, refresh_per_second=4, screen=True) as live:
        while True:
            dashboard.update()
            
            layout["header"].update(dashboard.render_header())
            layout["chemicals"].update(dashboard.render_chemicals())
            layout["genes"].update(dashboard.render_genes())
            layout["vm"].update(dashboard.render_vm())
            layout["task"].update(dashboard.render_task())
            layout["events"].update(dashboard.render_events())
            
            time.sleep(0.2)

if __name__ == "__main__":
    run_dashboard()
