
import torch
import os
from datetime import datetime
from torch.profiler import profile, record_function, ProfilerActivity

class ProfilerManager:

    def __init__(self, 
                 log_dir='./profile_logs',
                 activities=None,
                 profile_steps=5,
                 profile_memory=True,
                 record_shapes=True,
                 with_stack=True):
        
        self.log_dir = log_dir
        self.activities = activities or [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        self.profile_steps = profile_steps
        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.active_profiler = None
        self.is_profiling = False
        self.step_count = 0
        self.trace_exported = False  # Add this flag to track if trace was exported
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
    
    def start_profiling_session(self, epoch):
        """Start a new profiling session"""
        if self.is_profiling:
            return
            
        # Create a unique directory for this epoch's profile
        epoch_log_dir = os.path.join(self.log_dir, f'epoch_{epoch}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(epoch_log_dir, exist_ok=True)
        
        self.active_profiler = torch.profiler.profile(
            activities=self.activities,
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=self.profile_steps,
                repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(epoch_log_dir),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack
        )
        
        self.active_profiler.start()
        self.is_profiling = True
        self.step_count = 0
        self.current_epoch = epoch
        self.epoch_log_dir = epoch_log_dir
        self.trace_exported = False  # Reset trace export flag
        
        print(f"Profiling started for epoch {epoch}")
    
    def step(self):
        """Record a profiler step if profiling is active"""
        if not self.is_profiling:
            return
            
        self.active_profiler.step()
        self.step_count += 1
        
        # End profiling after the scheduled steps
        if self.step_count >= 2 + self.profile_steps:  # wait(1) + warmup(1) + active(profile_steps)
            self.end_profiling_session()
    
    def end_profiling_session(self):
        """End the current profiling session and print results"""
        if not self.is_profiling:
            return
            
        self.active_profiler.stop()
        
        # Print profile results
        print(f"\n--- Profile results for epoch {self.current_epoch} ---")
        print(self.active_profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # Export trace file for manual inspection - only if not already exported
        if not self.trace_exported:
            try:
                self.active_profiler.export_chrome_trace(os.path.join(self.epoch_log_dir, "trace.json"))
                self.trace_exported = True
            except RuntimeError as e:
                # Handle the case where trace is already saved
                if "Trace is already saved" in str(e):
                    print("Note: Trace was already exported automatically.")
                else:
                    # Re-raise if it's a different error
                    raise
        
        # Print forward/backward analysis
        #self.print_forward_backward_analysis()
        
        self.is_profiling = False
        self.active_profiler = None

    def print_forward_backward_analysis(self):
        """Print analysis of forward vs backward pass times"""
        if not self.active_profiler:
            return
            
        forward_time = sum(evt.cuda_time_total for evt in self.active_profiler.key_averages() 
                        if "forward" in evt.key.lower())
        backward_time = sum(evt.cuda_time_total for evt in self.active_profiler.key_averages() 
                        if "backward" in evt.key.lower())
        
        print(f"Forward time: {forward_time/1000:.2f}ms")
        print(f"Backward time: {backward_time/1000:.2f}ms")
        if forward_time > 0:
            print(f"Backward/Forward ratio: {backward_time/forward_time:.2f}x")
        
        # Print memory analysis if available
        if self.profile_memory:
            print("\nMemory Usage (Top 5):")
            print(self.active_profiler.key_averages().table(
                sort_by="self_cuda_memory_usage", row_limit=5))

# Add this context manager for conditional profiling
class ConditionalRecordFunction:
    def __init__(self, name, enable=True):
        self.name = name
        self.enable = enable
        
    def __enter__(self):
        if self.enable:
            self.record_ctx = record_function(self.name)
            self.record_ctx.__enter__()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            self.record_ctx.__exit__(exc_type, exc_val, exc_tb)
