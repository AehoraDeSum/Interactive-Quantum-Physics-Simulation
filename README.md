# A Real-Time Interactive Simulation Framework for Quantum Mechanics and Cosmic Ray Modeling

This open-source project provides a real-time computational physics simulation platform designed to model and visualize complex phenomena in **quantum mechanics** and **astroparticle physics**. The framework supports live experimentation, data visualization, and interactive parameter manipulation — making advanced physical systems accessible for education and scientific exploration.

---

## 🔬 Overview

Modern physics topics such as quantum behavior or cosmic ray interactions are abstract and non-intuitive. Traditional teaching approaches often fail to build conceptual understanding.

This project addresses that gap by providing:

- Real-time numerical simulation  
- A graphical user interface (PyQt5)  
- Computational physics models (Taichi + NumPy)  
- Data visualization tools (Matplotlib)  

---

## 🧠 Features

- 🌀 Quantum state evolution using the time-dependent Schrödinger equation  
- 🧲 Magnetic interaction modeling via the Zeeman effect  
- ☢️ Cosmic ray energy spectrum modeled using a validated power-law distribution  
- 📊 Real-time scientific visualization  
- 🖥️ GUI-based control environment  

---

## 📚 Scientific Background

### Quantum Mechanics Module

The simulation is based on the time-dependent Schrödinger equation:

\[
i\hbar \frac{\partial \psi}{\partial t} = \hat{H}\psi
\]

Where the Hamiltonian is defined as:

\[
\hat{H} = -\frac{\hbar^2}{2m}\nabla^2 + V(r) + \mu \cdot B
\]

The system supports ground states, excited states, superpositions, and spin-field interactions.

---

### Cosmic Ray Simulation Module

Cosmic ray behavior is modeled using an experimentally verified power-law spectrum:

\[
\frac{dN}{dE} \propto E^{-\gamma}, \quad \gamma = 2.7
\]

Detection follows a Poisson statistical framework to simulate event time distributions.

---

## 🧪 Running the Project

### **1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/Interactive-Physics-Simulation.git
cd Interactive-Physics-Simulation
