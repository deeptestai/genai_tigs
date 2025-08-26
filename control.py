from threading import Event

stop_flag = Event()

def stop_generation():
    stop_flag.set()
    return "🛑 Generation stopped by user."
