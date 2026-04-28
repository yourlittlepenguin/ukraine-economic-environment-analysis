# Ukraine Economic & Environmental Analysis

## Project Overview

This project is based on my Master's thesis focused on numerical methods for a dual dynamic balance model of interindustry ecological-economic interaction.
The practical part of the research includes analysis of real Ukrainian economic and environmental data across multiple years.

## Objectives

- Analyze multi-year Ukrainian economic indicators
- Explore environmental impact trends
- Study relationships between industries
- Build forecasting scenarios using mathematical modeling
- Visualize results using Python

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Data Analysis
- Forecasting Models

## Key Features

- Real-world Ukrainian data analysis
- Trend analysis across years
- Economic-environment interaction modeling
- Visual dashboards and charts
- Scenario evaluation

## Key Visualizations and Results

This section presents the main results of the economic-environmental system simulation under different scenarios, including shock propagation and sensitivity analysis.

---

### Sector Output Dynamics (2019–2024)

This chart shows the evolution of sectoral production output in the Ukrainian economy over time.

![Sector Output Dynamics](charts/sector_output_dynamics_2019_2024.png)

---

### Emissions under Shock Scenario (full-period shock)

This scenario simulates environmental emissions when a systemic shock is applied across all years of the observation period.

![Full Shock Emissions](charts/emissions_full_shock.png)

---

### Emissions under Single-Year Shock (2024 only)

This scenario isolates the impact of a shock introduced only in 2024, allowing comparison with gradual shock propagation.

![Single-Year Shock Emissions](charts/emissions_2024_shock.png)

---

### Sensitivity Analysis by Economic Sector

This analysis evaluates how individual economic sectors respond to changes in model parameters.

![Sector Sensitivity](charts/sensitivity_by_sector.png)

---

### Total System Sensitivity Across Scenarios

This chart summarizes the overall sensitivity of the economic-environmental system under different scenarios.

![Total Sensitivity](charts/total_system_sensitivity.png)

## Repository Structure

```text
data/           Raw and processed datasets
src/            Python scripts
charts/         Visualizations
docs/           Thesis PDF and project materials
