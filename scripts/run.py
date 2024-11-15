from simulation import Simulation

if __name__ == "__main__":
    sim = Simulation()
    sim.initialize_simulation()
    try:
        sim.run()
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        sim.cleanup()
