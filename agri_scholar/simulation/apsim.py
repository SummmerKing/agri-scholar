"""Thin wrapper around an APSIM NextGen (.apsimx) JSON file.

Exposes the operations in the action space (sowing, tillage and mulch, basal DAP, urea top-dressing,
irrigation), runs the simulation and reads back yield, stress and the sustainability indicators.
"""
import json
import os
import shutil
import subprocess
import sqlite3
import logging
import pandas as pd

log = logging.getLogger("agri_sigra")

# apsimNGpy scans $HOME on import, so the Models executable is called directly
ApsimModel = None

# Growth stages counted as "reproductive" for phenology-partitioned stress (repro_* in Eq. usage).
REPRODUCTIVE_STAGES = {"FlagLeaf", "Flowering", "StartGrainFill", "EndGrainFill",
                        "Maturity", "GrainFilling", "Reproductive"}


class ApsimController:
    def __init__(self, base_file: str, working_file: str):
        self.base_file = base_file
        self.working_file = working_file
        os.makedirs(os.path.dirname(self.working_file), exist_ok=True)
        shutil.copy(self.base_file, self.working_file)
        with open(self.working_file, "r") as f:
            self.data = json.load(f)

    def _find_node(self, root: dict, name: str):
        if root.get("Name") == name:
            return root
        for child in root.get("Children", []):
            found = self._find_node(child, name)
            if found:
                return found
        return None

    def _save(self):
        if isinstance(self.data, dict) and "Version" in self.data:
            self.data["Version"] = 168
        with open(self.working_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def _inject_manager(self, name: str, code_lines: list, parameters: dict):
        """Generic helper: injects a C# Manager script into the Field/Zone node, idempotently."""
        script = {
            "$type": "Models.Manager, Models", "Name": name, "CodeArray": code_lines,
            "Parameters": [{"Key": k, "Value": str(v)} for k, v in parameters.items()],
            "Enabled": True, "ReadOnly": False,
        }

        def inject(node):
            if isinstance(node, dict):
                if node.get("Name") == "Field" or "Zone" in node.get("$type", ""):
                    for child in node.get("Children", []):
                        if child.get("Name") == name:
                            return True
                    node.setdefault("Children", []).append(script)
                    return True
                for child in node.get("Children", []):
                    if inject(child):
                        return True
            return False

        return inject(self.data)

    def set_clock_and_weather(self, year: int, met_file: str):
        clock = self._find_node(self.data, "Clock")
        if clock:
            clock["Start"] = f"{year}-01-01T00:00:00"
            clock["End"] = f"{year}-12-31T00:00:00"
        weather = self._find_node(self.data, "Weather")
        if weather:
            weather["FileName"] = met_file
        self._save()

    def disable_default_managers(self):
        """Disable the fixed fertiliser/irrigation managers in the base file so the agents control all actions."""
        def disable(node):
            if isinstance(node, dict):
                if node.get("Name") in ("Fertiliser1", "Irrigation at fixed intervals"):
                    node["Enabled"] = False
                for v in node.values():
                    disable(v)
            elif isinstance(node, list):
                for item in node:
                    disable(item)
        disable(self.data)
        self._save()

    def set_location_soil_profile(self, location: str, soil_profiles: dict):
        """`soil_profiles` maps location name -> physics dict (see agri_scholar/config.py)."""
        p = soil_profiles.get(location, next(iter(soil_profiles.values())))
        found = 0

        def update(node):
            nonlocal found
            if isinstance(node, dict):
                if node.get("$type", "").startswith("Models.Soils.Soil"):
                    found += 1
                    node.update({"ASCOrder": p["order"], "SoilType": p["type"], "Site": location,
                                 "NearestTown": location, "Latitude": p["lat"], "Longitude": p["lon"]})
                    physical = next((c for c in node.get("Children", [])
                                      if c.get("$type", "").startswith("Models.Soils.Physical")), None)
                    if physical:
                        physical.update({"BD": p["bd"], "AirDry": p["airdry"], "LL15": p["ll15"],
                                          "DUL": p["dul"], "SAT": p["sat"], "KS": p["ks"]})
                        soil_crop = next((c for c in physical.get("Children", [])
                                           if c.get("$type", "").startswith("Models.Soils.SoilCrop")), None)
                        if soil_crop:
                            soil_crop["LL"] = p["ll15"]
                    organic = next((c for c in node.get("Children", [])
                                     if c.get("$type", "").startswith("Models.Soils.Organic")), None)
                    if organic:
                        organic["Carbon"] = p["carbon"]
                    water = next((c for c in node.get("Children", [])
                                   if c.get("$type", "").startswith("Models.Soils.Water")), None)
                    if water:
                        water["InitialValues"] = p["dul"]
                for v in node.values():
                    update(v)
            elif isinstance(node, list):
                for item in node:
                    update(item)

        update(self.data)
        if found:
            self._save()
        return found > 0

    def set_sowing_rules(self, start_date: str, end_date: str, population: float, cultivar: str):
        def update(node):
            if isinstance(node, dict):
                if node.get("Name") == "Sow using a variable rule":
                    for p in node.get("Parameters", []):
                        if p["Key"] == "Population": p["Value"] = str(population)
                        if p["Key"] == "CultivarName": p["Value"] = str(cultivar)
                        if p["Key"] == "StartDate": p["Value"] = str(start_date)
                        if p["Key"] == "EndDate": p["Value"] = str(end_date)
                        if p["Key"] == "MustSow": p["Value"] = "yes"
                        if p["Key"] == "MinESW": p["Value"] = "0"
                        if p["Key"] == "MinRain": p["Value"] = "0"
                        if p["Key"] == "RainDays": p["Value"] = "0"
                    return True
                for v in node.values():
                    if update(v): return True
            elif isinstance(node, list):
                for item in node:
                    if update(item): return True
            return False

        if update(self.data):
            self._save()
            return

        # no sowing rule in the file: add a dated sowing manager
        self._inject_manager(
            "SowOnDate",
            ["using Models.Core;", "using Models.PMF;", "using System;",
             "namespace Models { [Serializable] public class Script : Model {",
             "   [Link] Clock Clock; [Link] Plant Maize;",
             "   public string StartDate { get; set; } public double Population { get; set; } public string CultivarName { get; set; }",
             "   [EventSubscribe(\"DoManagement\")] private void OnDoManagement(object sender, EventArgs e) {",
             "       if (APSIM.Shared.Utilities.DateUtilities.DatesEqual(StartDate, Clock.Today) && Maize != null) {",
             "           Maize.Sow(cultivar: CultivarName, population: Population, depth: 50.0, rowSpacing: 600.0); } }",
             "} }"],
            {"StartDate": start_date, "Population": population, "CultivarName": cultivar},
        )
        self._save()

    def set_mulch(self, amount_kg_ha: float):
        def update(node):
            if isinstance(node, dict):
                if node.get("$type", "").startswith("Models.Surface.SurfaceOrganicMatter"):
                    node["InitialResidueMass"] = float(amount_kg_ha)
                    return True
                for v in node.values():
                    if update(v): return True
            elif isinstance(node, list):
                for item in node:
                    if update(item): return True
            return False
        ok = update(self.data)
        if ok:
            self._save()
        return ok

    def add_tillage_event(self, date_str: str, fraction_incorporated: float = 0.5, depth_mm: float = 150.0):
        return self._inject_manager(
            f"Tillage on {date_str}",
            ["using Models.Core;", "using Models.Surface;", "using System;",
             "namespace Models { [Serializable] public class Script : Model {",
             "   [Link] Clock Clock; [Link] SurfaceOrganicMatter SOM;",
             "   public string TillageDate { get; set; } public double Fraction { get; set; } public double Depth { get; set; }",
             "   [EventSubscribe(\"DoManagement\")] private void OnDoManagement(object sender, EventArgs e) {",
             "       if (APSIM.Shared.Utilities.DateUtilities.DatesEqual(TillageDate, Clock.Today)) { if (SOM != null) SOM.Incorporate(Fraction, Depth); } }",
             "} }"],
            {"TillageDate": date_str, "Fraction": fraction_incorporated, "Depth": depth_mm},
        ) and self._save() or True

    def add_fertilizer(self, date_str: str, amount_kg: float, fert_type: str = "UreaN", depth_mm: float = 50.0):
        """Applies any APSIM fertiliser type (UreaN, DAP, map, NO3N, NH4N, ...) on a given date."""
        name = f"{fert_type} {amount_kg:.1f}kg on {date_str}"
        ok = self._inject_manager(
            name,
            ["using Models.Core;", "using System;",
             "namespace Models { [Serializable] public class Script : Model {",
             "   [Link] Clock Clock; [Link] Fertiliser Fert;",
             "   public string FertDate { get; set; } public double Amount { get; set; }",
             "   public double Depth { get; set; } public string FertType { get; set; }",
             "   [EventSubscribe(\"DoManagement\")] private void OnDoManagement(object sender, EventArgs e) {",
             "       if (APSIM.Shared.Utilities.DateUtilities.DatesEqual(FertDate, Clock.Today) && Fert != null) {",
             "           Fert.Apply(Amount: Amount, Type: FertType, Depth: Depth); } }",
             "} }"],
            {"FertDate": date_str, "Amount": amount_kg, "Depth": depth_mm, "FertType": fert_type},
        )
        if ok:
            self._save()
        return ok

    def apply_irrigation(self, date_str: str, amount_mm: float):
        name = f"Irrigate {amount_mm:.1f}mm on {date_str}"
        ok = self._inject_manager(
            name,
            ["using Models.Core;", "using Models.WaterModel;", "using System;",
             "namespace Models { [Serializable] public class Script : Model {",
             "   [Link] Clock Clock; [Link] Irrigation Irrig;",
             "   public string IrrigDate { get; set; } public double Amount { get; set; }",
             "   [EventSubscribe(\"DoManagement\")] private void OnDoManagement(object sender, EventArgs e) {",
             "       if (APSIM.Shared.Utilities.DateUtilities.DatesEqual(IrrigDate, Clock.Today)) { if (Irrig != null) Irrig.Apply(Amount); } }",
             "} }"],
            {"IrrigDate": date_str, "Amount": amount_mm},
        )
        if ok:
            self._save()
        return ok

    def _inject_daily_report_variables(self):
        """Forces the Report node to log daily crop-stress, phenology, and sustainability variables."""
        needed_vars = [
            "[Maize].Leaf.Fw", "[Maize].Leaf.Fn",
            "[Leaf].NDemand.Total", "[Stem].NDemand.Total", "[Root].NDemand.Total", "[Grain].NDemand.Total",
            "[WaterBalance].Drainage", "[WaterBalance].LeachNO3", "[WaterBalance].LeachNH4",
            "[Maize].Phenology.CurrentStageName", "[SurfaceOrganicMatter].Wt as Mulch_Wt",
            "sum([Soil].Physical.SWmm) as Total_SW_mm",
            "sum([Soil].Physical.DULmm) as Total_DUL_frac",
            "sum([Soil].Physical.SATmm) as Total_SAT_frac",
            "[Fertiliser].NitrogenApplied",
            "sum([Soil].Organic.Carbon) as Total_SOC_pct",
        ]

        def update(node):
            if isinstance(node, dict):
                if node.get("$type", "").startswith("Models.Report"):
                    variables = node.get("VariableNames", [])
                    for v in needed_vars:
                        if v not in variables:
                            variables.append(v)
                    node["VariableNames"] = variables
                    events = node.get("EventNames", [])
                    if "[Clock].EndOfDay" not in events:
                        events.append("[Clock].EndOfDay")
                    node["EventNames"] = events
                    return True
                for v in node.values():
                    if update(v): return True
            elif isinstance(node, list):
                for item in node:
                    if update(item): return True
            return False

        update(self.data)
        self._save()

    def _execute(self, report_name: str = "Report") -> pd.DataFrame:
        if ApsimModel is not None:
            try:
                model = ApsimModel(self.working_file, read_from_string=False)
                model.run(report_name=report_name)
                df = model.results.copy()
                try:
                    model.clean_up(db=True)
                except Exception:
                    pass
                if len(df) > 0:
                    return df
            except Exception as e:
                log.warning(f"apsimNGpy run failed, falling back to Models.exe: {e}")

        models_exe = (shutil.which("Models.exe") or shutil.which("Models")
                      or os.environ.get("APSIM_MODELS_EXE"))
        if models_exe and os.path.exists(models_exe):
            try:
                with open(models_exe, "r") as f:
                    content = f.read().strip()
                if content.startswith("exec dotnet ") or content.startswith("dotnet "):
                    dll_path = content.replace("exec ", "").replace("dotnet ", "").replace('"$@"', "").strip()
                    if os.path.exists(dll_path):
                        subprocess.run(["dotnet", dll_path, self.working_file], capture_output=True, text=True)
                    else:
                        subprocess.run(["bash", models_exe, self.working_file], capture_output=True, text=True)
                else:
                    subprocess.run([models_exe, self.working_file], capture_output=True, text=True)
            except (UnicodeDecodeError, ValueError):
                subprocess.run([models_exe, self.working_file], capture_output=True, text=True)

            db_file = self.working_file.replace(".apsimx", ".db")
            if os.path.exists(db_file):
                try:
                    conn = sqlite3.connect(db_file)
                    df = pd.read_sql_query(f"SELECT * FROM {report_name}", conn)
                    conn.close()
                    return df
                except Exception as e:
                    log.warning(f"Could not read APSIM sqlite output: {e}")
        return pd.DataFrame()

    def run_and_get_yield(self, sow_doy: int = None, total_weeks: int = 18) -> dict:
        """Run the simulation and return yield, harvest index, water/N stress, drainage, leaching and I_wl, I_no, I_cl."""
        self._inject_daily_report_variables()
        for ext in ("", "-shm", "-wal"):
            target = self.working_file.replace(".apsimx", ".db") + ext
            if os.path.exists(target):
                try:
                    os.remove(target)
                except OSError:
                    pass

        try:
            df = self._execute(report_name="Report")
            if df.empty:
                return {"status": "failed", "yield_kg_ha": 0}

            wt_col = next((c for c in df.columns if "AboveGround.Wt" in c), None)
            if not wt_col:
                return {"status": "failed", "yield_kg_ha": 0}
            df[wt_col] = pd.to_numeric(df[wt_col], errors="coerce").fillna(0)
            df_crop = df[df[wt_col] > 0].copy()
            if df_crop.empty:
                return {"status": "failed", "yield_kg_ha": 0}

            yield_col = next((c for c in df_crop.columns if "MaizeYieldkgha" in c or "Grain.Wt" in c), None)
            if yield_col:
                df_crop[yield_col] = pd.to_numeric(df_crop[yield_col], errors="coerce").fillna(0)
                yield_kg_ha = float(df_crop[yield_col].max()) if "MaizeYieldkgha" in yield_col else float(df_crop[yield_col].max()) * 10.0
            else:
                yield_kg_ha = 0.0

            final_biomass = df_crop[wt_col].max()
            final_grain_wt = (yield_kg_ha / 10.0) if yield_kg_ha > 0 else 0.0
            harvest_index = (final_grain_wt / final_biomass) if final_biomass > 0 else 0.0

            fw_col = next((c for c in df_crop.columns if "Leaf.Fw" in c), None)
            fn_col = next((c for c in df_crop.columns if "Leaf.Fn" in c), None)
            drainage_col = next((c for c in df_crop.columns if "Drainage" in c), None)
            stage_col = next((c for c in df_crop.columns if "CurrentStageName" in c or "StageName" in c), None)

            veg_w = veg_n = repro_w = repro_n = 0.0
            veg_days = repro_days = 0
            weekly_stress = {}

            if stage_col and fw_col and fn_col:
                for _, row in df_crop.iterrows():
                    stage = str(row[stage_col]).strip()
                    fw_deficit = max(0.0, 1.0 - float(row[fw_col]))
                    fn_deficit = max(0.0, 1.0 - float(row[fn_col]))
                    if stage in REPRODUCTIVE_STAGES:
                        repro_w += fw_deficit; repro_n += fn_deficit; repro_days += 1
                    else:
                        veg_w += fw_deficit; veg_n += fn_deficit; veg_days += 1
            elif fw_col and fn_col:
                n_days = len(df_crop)
                split = int(n_days * 0.6)
                veg_w = (1.0 - df_crop[fw_col].iloc[:split]).clip(lower=0).sum()
                repro_w = (1.0 - df_crop[fw_col].iloc[split:]).clip(lower=0).sum()
                veg_n = (1.0 - df_crop[fn_col].iloc[:split]).clip(lower=0).sum()
                repro_n = (1.0 - df_crop[fn_col].iloc[split:]).clip(lower=0).sum()
                veg_days, repro_days = split, n_days - split

            total_drainage = float(df_crop[drainage_col].sum()) if drainage_col else 0.0

            if sow_doy is not None and fw_col and fn_col:
                date_col = next((c for c in df_crop.columns if "Date" in c or "Clock.Today" in c), None)
                dfc = df_crop.copy()
                if date_col:
                    dfc["_doy"] = pd.to_datetime(dfc[date_col], errors="coerce").dt.dayofyear
                else:
                    dfc["_doy"] = range(1, len(dfc) + 1)
                for wk in range(total_weeks):
                    lo, hi = sow_doy + wk * 7, sow_doy + wk * 7 + 7
                    rows = dfc[(dfc["_doy"] >= lo) & (dfc["_doy"] < hi)]
                    if len(rows):
                        weekly_stress[f"week_{wk}"] = {
                            "water_stress": round(float((1.0 - rows[fw_col].astype(float)).clip(lower=0).sum()), 3),
                            "n_stress": round(float((1.0 - rows[fn_col].astype(float)).clip(lower=0).sum()), 3),
                            "days": len(rows),
                        }
                    else:
                        weekly_stress[f"week_{wk}"] = {"water_stress": 0.0, "n_stress": 0.0, "days": 0}

            leach_no3 = next((c for c in df_crop.columns if "LeachNO3" in c), None)
            leach_nh4 = next((c for c in df_crop.columns if "LeachNH4" in c), None)
            total_leach = (df_crop[leach_no3].astype(float).sum() if leach_no3 else 0.0) + \
                          (df_crop[leach_nh4].astype(float).sum() if leach_nh4 else 0.0)

            sw_col = next((c for c in df_crop.columns if "Total_SW_mm" in c), None)
            dul_col = next((c for c in df_crop.columns if "Total_DUL" in c), None)
            sat_col = next((c for c in df_crop.columns if "Total_SAT" in c), None)
            napp_col = next((c for c in df_crop.columns if "NitrogenApplied" in c), None)
            soc_col = next((c for c in df_crop.columns if "Total_SOC" in c), None)

            if sw_col and dul_col and sat_col:
                denom = (df_crop[sat_col].astype(float) - df_crop[dul_col].astype(float)).replace(0, pd.NA)
                wl_ratio = ((df_crop[sw_col].astype(float) - df_crop[dul_col].astype(float)) / denom)
                I_wl = float(wl_ratio.clip(lower=0).fillna(0).mean())   # Eq. 9
            else:
                I_wl = 0.0

            n_demand_cols = [c for c in [
                next((c for c in df_crop.columns if "Leaf.NDemand.Total" in c), None),
                next((c for c in df_crop.columns if "Stem.NDemand.Total" in c), None),
                next((c for c in df_crop.columns if "Root.NDemand.Total" in c), None),
                next((c for c in df_crop.columns if "Grain.NDemand.Total" in c), None),
            ] if c]
            if napp_col and n_demand_cols:
                daily_demand = df_crop[n_demand_cols].astype(float).sum(axis=1)
                oversupply = (df_crop[napp_col].astype(float).fillna(0) - daily_demand).clip(lower=0)
                I_no = float(oversupply.mean())                        # Eq. 10
            else:
                I_no = 0.0

            if soc_col and len(df_crop) > 0:
                I_cl = max(0.0, float(df_crop[soc_col].iloc[0]) - float(df_crop[soc_col].iloc[-1]))  # Eq. 13
            else:
                I_cl = 0.0

            return {
                "status": "success",
                "yield_kg_ha": round(yield_kg_ha),
                "harvest_index": round(harvest_index, 4),
                "peak_biomass_g_m2": round(final_biomass, 2),
                "veg_water_stress": round(veg_w, 3), "veg_n_stress": round(veg_n, 3), "veg_days": veg_days,
                "repro_water_stress": round(repro_w, 3), "repro_n_stress": round(repro_n, 3), "repro_days": repro_days,
                "crop_lifespan_days": len(df_crop),
                "weekly_stress": weekly_stress,
                "total_drainage_mm": round(total_drainage, 2),      # I_dd, Eq. 11
                "total_leach_no3_nh4": round(total_leach, 2),       # I_nl, Eq. 12
                "waterlogging_index": round(I_wl, 5),                # I_wl, Eq. 9
                "n_oversupply_index": round(I_no, 3),                 # I_no, Eq. 10
                "soc_loss_index": round(I_cl, 4),                      # I_cl, Eq. 13
            }
        except Exception as e:
            log.exception(f"APSIM run error: {e}")
            return {"status": "error", "yield_kg_ha": 0}
