# CUAV X7 Pro Flight Controller - Benewake TF03 (4-20mA) Integration & Wiring Guide

**Document Version:** 1.0  
**Target Hardware:** CUAV X7 Pro / CUAV X7+ Autopilot  
**Sensor Model:** Benewake TF03-180 (4-20mA Analog Current Output Model, 5V-24V DC)  
**Firmware:** ArduCopter 4.7.1 (AP_RangeFinder_Analog)  
**Date:** September 2026  

---

## 1. Executive Summary & Working Principle

The **Benewake TF03-4-20mA** is an industrial-grade Pulsed Time-of-Flight (PToF) laser LiDAR that measures distances up to **180 meters**. 

Unlike standard drone LiDARs that transmit serial UART data, the 4-20mA version outputs distance as a continuous **analog current loop (4mA to 20mA)** on its **Green wire (Pin 3)**:
* **4 mA** corresponds to **0.0 meters** (or minimum range).
* **20 mA** corresponds to **180.0 meters** (maximum range).
* The UART interface (Brown and Blue wires) on this specific model is reserved for manufacturer configuration and does not output real-time detection data.

To connect this 4-20mA current loop to the **CUAV X7 Pro Flight Controller**, the current signal must be converted to an **analog voltage (0.6V to 3.0V)** and fed into **Port 09: ADC (Pin 2)**. ArduPilot native driver `RNGFND1_TYPE = 1` (Analog) samples this voltage at 100Hz to derive real-time altitude.

---

## 2. Bill of Materials (Required Items / Saman Ki List)

To complete this installation, you will need the following components:

| Item No. | Component Name | Specification | Purpose | Approx. Cost |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Signal Sampling Resistor** *(Option A)* | **$150\ \Omega$ (or $250\ \Omega$)**, 1/4W or 1/2W, 1% or 0.1% tolerance | Converts 4-20mA current to 0.6V-3.0V voltage | ₹5 - ₹10 |
| **OR** | **4-20mA to 0-3.3V Converter Module** *(Option B)* | LM358-based Current-to-Voltage Converter with Zero/Span pots | Ready-made module with screw terminals | ₹150 - ₹250 |
| **2** | **JST-GH 5-Pin Cable** | 1.25mm pitch JST-GH connector | Plugs into CUAV X7 Pro Port 09 (ADC) | Standard accessory |
| **3** | **Heat Shrink Tubing / Tape** | 2mm - 4mm diameter shrink tube | Electrical insulation for solder joints | ₹20 |
| **4** | **Power Connection** | 5V to 24V DC (CUAV 5V rail or Drone BEC) | Supplies power to TF03 | Existing harness |

---

## 3. Visual Wiring Diagram & Schematic

Below is the complete, high-resolution electrical schematic showing the pin-to-pin wiring between the **Benewake TF03**, the **150-Ohm sampling resistor**, and the **CUAV X7 Pro Port 09 (ADC)**:

![TF03 Wiring Schematic](/home/abc/Desktop/DRONE/tf03_wiring_diagram.png)

---

## 4. Pin-by-Pin Connection Table

### A. Benewake TF03 Connector (Molex 7-Pin Connector)
*From Official Benewake TF03 User Manual Table 3:*

| Pin | Wire Color | Signal Name | Connection Destination |
| :---: | :---: | :--- | :--- |
| **1** | **RED** | VCC (+5V to +24V DC) | Connect to **CUAV Port 09 Pin 1 (+5V)** or Drone 12V BEC |
| **2** | **WHITE** | N/A (Not Connected) | **LEAVE DISCONNECTED** (Insulate with tape) |
| **3** | **GREEN** | **Analog Signal (4-20mA)** | Connect to **CUAV Port 09 Pin 2 (ADC)** + Resistor |
| **4** | **N/A** | Empty Slot | N/A |
| **5** | **BLUE** | UART RxD (Config only) | **LEAVE DISCONNECTED** (Insulate with tape) |
| **6** | **BROWN** | UART TxD (Config only) | **LEAVE DISCONNECTED** (Insulate with tape) |
| **7** | **BLACK** | **GND (Power & Signal Ground)** | Connect to **CUAV Port 09 Pin 4 (GND)** |

---

### B. CUAV X7 Pro Port 09: ADC Connector (JST-GH 5-Pin)
*From CUAV X7 Family Pinout Specification:*

| Pin | Label | Function | Connected To |
| :---: | :---: | :--- | :--- |
| **1** | **5V+** | +5V DC Regulated Power Output | TF03 **Red Wire** (Pin 1) |
| **2** | **ADC** | Analog Voltage Input (0V to 3.3V) | TF03 **Green Wire** (Pin 3) + $150\ \Omega$ Resistor Lead |
| **3** | **ADC2** | Secondary Analog / RSSI | Not Used |
| **4** | **GND** | Signal Ground | TF03 **Black Wire** (Pin 7) + Resistor Ground Lead |
| **5** | **GND** | System Ground | Optional Ground |

---

## 5. Circuit Assembly Instructions (Step-by-Step)

### Option A: Using a Simple 150-Ohm Resistor (Simplest & Lightest)
1. **Prepare the Resistor:** Take a $150\ \Omega$ (0.25W or 0.5W) metal film precision resistor.
2. **Solder Lead 1:** Connect one lead of the $150\ \Omega$ resistor to the **Green wire** of the TF03 and Pin 2 of the JST-GH cable.
3. **Solder Lead 2:** Connect the other lead of the $150\ \Omega$ resistor to the **Black wire** of the TF03 and Pin 4 (GND) of the JST-GH cable.
4. **Insulate:** Slide heat-shrink tubing over the resistor and solder joints, then heat to seal.
5. **Power:** Solder the **Red wire** of the TF03 to Pin 1 (+5V) of the JST-GH cable.
6. **Plug In:** Insert the JST-GH 5-pin connector firmly into **Port 09 (ADC)** of the CUAV X7 Pro.

### Option B: Using a 4-20mA to Voltage Converter Module
1. Connect TF03 **Red (+5V)** and **Black (GND)** to the power supply.
2. Connect TF03 **Green wire** to the **Current Input (I+)** terminal of the module.
3. Connect the module **GND (I-)** to the system ground.
4. Connect the module **Voltage Output (V_OUT)** to **CUAV Port 09 Pin 2 (ADC)**.
5. Connect the module **GND** to **CUAV Port 09 Pin 4 (GND)**.

---

## 6. Mathematical Scaling & Calibration

According to Ohm\x27s Law:
$$V = I \times R$$

With a $150\ \Omega$ sampling resistor:
* At minimum distance ($0\text{ m}$, output $4\text{ mA} = 0.004\text{ A}$):
  $$V_{min} = 0.004\text{ A} \times 150\ \Omega = \mathbf{0.60\text{ Volts}}$$
* At maximum distance ($180\text{ m}$, output $20\text{ mA} = 0.020\text{ A}$):
  $$V_{max} = 0.020\text{ A} \times 150\ \Omega = \mathbf{3.00\text{ Volts}}$$
* **Voltage Span:** $\Delta V = 3.00\text{V} - 0.60\text{V} = 2.40\text{ Volts}$.
* **Distance Span:** $\Delta D = 180.0\text{ meters}$.
* **ArduPilot Scaling:**
  $$\text{SCALING} = \frac{\Delta D}{\Delta V} = \frac{180.0}{2.40} = \mathbf{75.0\text{ meters/volt}}$$
* **ArduPilot Offset:**
  $$\text{OFFSET} = - (V_{min} \times \text{SCALING}) = - (0.60 \times 75.0) = \mathbf{-45.0\text{ meters}}$$

---

## 7. Mission Planner Parameter Configuration

Connect your drone to Mission Planner, navigate to **`CONFIG`** $\rightarrow$ **`Full Parameter List`**, and configure the following parameters:

| Parameter Name | Value | Meaning / Rationale |
| :--- | :---: | :--- |
| **`RNGFND1_TYPE`** | **`1`** | **Analog Rangefinder** (Samples voltage on ADC pin) |
| **`RNGFND1_PIN`** | **`103`** | CUAV X7 Pro Analog ADC Port 09 pin number |
| **`RNGFND1_SCALING`**| **`75.0`** | Conversion factor ($75.0\text{ meters per volt}$) |
| **`RNGFND1_OFFSET`** | **`-45.0`**| Voltage offset (calibrates $0.60\text{V}$ to $0.0\text{m}$) |
| **`RNGFND1_MIN`** | **`0.10`** | Minimum valid distance ($10\text{ cm}$) |
| **`RNGFND1_MAX`** | **`180.0`**| Maximum valid distance ($180.0\text{ meters}$) |
| **`RNGFND1_GNDCLR`** | **`0.15`** | Distance from sensor to floor when resting on landing gear ($15\text{ cm}$) |
| **`RNGFND1_ORIENT`** | **`25`** | **Downward-Facing** (Orientation: Pitch 270 / Down) |
| **`EK3_SRC1_POSZ`** | **`2.0`** | Use Rangefinder as primary EKF3 Altitude source for low hover |
| **`EK3_RNG_USE_HGT`**| **`70.0`** | Use Rangefinder up to 70% of maximum range |

Click **`Write Params`** and reboot the flight controller.

---

## 8. Verification & Live Bench Testing Procedure

1. **Power On:** Connect the drone battery.
2. **Open Mission Planner:** Click **CONNECT** (Baud: 115200).
3. **Check Live Voltage & Distance:**
   - Go to **`FLIGHT DATA`** $\rightarrow$ **`Status`** tab.
   - Look for **`sonarvoltage`**: When pointing at a nearby wall or floor ($0.5\text{m} - 1.0\text{m}$), voltage should read approximately **$0.61\text{V} - 0.65\text{V}$**.
   - Look for **`sonarrange`**: It should accurately display the distance in meters.
4. **Hand Obstacle Test:** Place your hand $50\text{ cm}$ in front of the TF03 lens. `sonarrange` should instantly update to `0.50m`. Move hand to $1.0\text{m}$, reading should smoothly follow to `1.00m`.
5. **Pre-Arm Check:** Ensure all Pre-Arm checks pass with no `Rangefinder 1: No Data` errors.

---

*Generated by Antigravity Autonomous Flight Systems - Pair Programming Engineering Support.*
