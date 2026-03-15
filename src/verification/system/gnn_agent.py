"""GNN-based heading predictor agent for the verification system.

Wraps a trained ``GATHeadingPredictor`` or ``BaseGNN`` model as an
``AgentModel``, handling observation to graph conversion and heading bin
decoding.  Use ``GNNAgent.from_checkpoint()`` to load a trained model.

Graph construction mirrors the training-time ``AircraftGraphDataset`` exactly:

Node features (11-D per aircraft, plus one airport hub node at the end):
    [lat_norm, lon_norm, time_offset_norm,
     sin(heading), cos(heading),
     velocity_norm, altitude_norm, vertrate_norm,
     dist_to_airport_norm,
     sin(bearing_to_airport), cos(bearing_to_airport)]

Edge features (5-D):
    [dist_nm, sin(bearing), cos(bearing),
     sin(rel_heading), cos(rel_heading)]

Airport-aircraft edges use rel_heading = 0.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch_geometric.data import Data

from aircraft_heading_gnn.models.base_gnn import BaseGNN, GATHeadingPredictor
from aircraft_heading_gnn.utils.adsb_features import get_azimuth_to_point, get_distance_nm
from verification.system.actions import Action
from verification.system.observations import Observation
from verification.system.system import AgentModel

# Constants from AircraftGraphDataset
_TERMINAL_RADIUS_NM  = 180.0
_TERMINAL_RADIUS_DEG = _TERMINAL_RADIUS_NM / 60.0   # degrees
_MAX_EDGE_DIST_NM    = 35.0
_FT_TO_M             = 0.3048
_FPM_TO_MS           = _FT_TO_M / 60.0
_MPS_TO_KTS          = 1.0 / 0.51444444  # training velocity column is in knots

class GNNAgent(AgentModel):
    """Agent that wraps a trained GNN heading predictor.

    Args:
        model:            Trained GNN model.
        airport_lat:      Airport reference latitude (degrees).
        airport_lon:      Airport reference longitude (degrees).
        device:           Torch device string.
        num_classes:      Number of heading bins (default 72 for 5° bins).
        heading_bin_size: Degrees per bin (default 5.0).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        airport_lat: float,
        airport_lon: float,
        device: str = "cpu",
        num_classes: int = 72,
        heading_bin_size: float = 5.0,
    ) -> None:
        self.model            = model.to(device)
        self.model.eval()
        self.airport_lat      = airport_lat
        self.airport_lon      = airport_lon
        self.device           = device
        self.num_classes      = num_classes
        self.heading_bin_size = heading_bin_size

    # ------------------------------------------------------------------
    # AgentModel API
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Compute a heading command from *observation* using the GNN."""
        graph = self._observation_to_graph(observation)

        with torch.no_grad():
            graph = graph.to(self.device)
            logits = self.model(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                None,
            )

        # Ego is always node 0; ignore the airport hub node at the end
        ego_logits = logits[0]
        probs      = torch.softmax(ego_logits, dim=0)

        # Circular mean over the softmax distribution — gives sub-bin-resolution
        # heading rather than snapping to the argmax bin centre.
        bin_centres = torch.arange(self.num_classes, dtype=torch.float32, device=self.device)
        bin_centres = torch.deg2rad(bin_centres * self.heading_bin_size + self.heading_bin_size / 2)
        sin_mean = float((probs * torch.sin(bin_centres)).sum())
        cos_mean = float((probs * torch.cos(bin_centres)).sum())
        predicted_heading_rad = math.atan2(sin_mean, cos_mean) % (2 * math.pi)

        predicted_class = int(torch.argmax(probs).item())
        return Action(
            heading_command=predicted_heading_rad,
            metadata={
                "predicted_class": predicted_class,
                "confidence": float(probs[predicted_class].item()),
            },
        )

    # ------------------------------------------------------------------
    # Graph construction  (mirrors AircraftGraphDataset exactly)
    # ------------------------------------------------------------------

    def _observation_to_graph(self, observation: Observation) -> Data:
        """Convert *observation* to a PyG ``Data`` matching the training graph format."""
        all_aircraft = [observation.ego_state] + list(observation.traffic_states)
        n = len(all_aircraft)

        # --- collect per-aircraft lat/lon (stored in metadata by _build_state) ---
        lats, lons, headings_rad, vels, alts_m, vrs_ms = [], [], [], [], [], []
        for ac in all_aircraft:
            meta = ac.metadata or {}
            lat  = meta.get("lat")
            lon  = meta.get("lon")
            if lat is None or lon is None:
                # Fallback: reconstruct from ENU position (less accurate)
                cos_lat = math.cos(math.radians(self.airport_lat))
                lat = ac.position[1] / 111_320.0 + self.airport_lat
                lon = ac.position[0] / (111_320.0 * cos_lat) + self.airport_lon
            alt_m = float(meta.get("altitude", 0.0)) * _FT_TO_M
            vr_ms = float(meta.get("vertrate",  0.0)) * _FPM_TO_MS
            lats.append(float(lat))
            lons.append(float(lon))
            headings_rad.append(float(ac.heading))
            vels.append(float(np.linalg.norm(ac.velocity[:2])) * _MPS_TO_KTS)   # horizontal ground speed → knots
            alts_m.append(alt_m)
            vrs_ms.append(vr_ms)

        # --- node features ---
        node_features = []
        for i in range(n):
            lat_n  = (lats[i] - self.airport_lat) / _TERMINAL_RADIUS_DEG
            lon_n  = (lons[i] - self.airport_lon) / _TERMINAL_RADIUS_DEG
            hdg    = headings_rad[i]
            d_ap   = get_distance_nm(lats[i], lons[i], self.airport_lat, self.airport_lon)
            b_ap   = math.radians(get_azimuth_to_point(lats[i], lons[i], self.airport_lat, self.airport_lon))
            node_features.append([
                lat_n,
                lon_n,
                0.0,                          # time_offset_norm = 0 at inference
                math.sin(hdg),
                math.cos(hdg),
                vels[i] / 200.0,
                alts_m[i] / 3000.0,
                vrs_ms[i] / 10.0,
                d_ap / _TERMINAL_RADIUS_NM,
                math.sin(b_ap),
                math.cos(b_ap),
            ])

        # Airport hub node
        node_features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        airport_idx = n

        # --- edges ---
        edge_index_list, edge_attr_list = [], []

        # Aircraft–aircraft
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dist = get_distance_nm(lats[i], lons[i], lats[j], lons[j])
                if dist > _MAX_EDGE_DIST_NM:
                    continue
                b_ij       = math.radians(get_azimuth_to_point(lats[i], lons[i], lats[j], lons[j]))
                rel_hdg    = headings_rad[j] - headings_rad[i]
                edge_index_list.append([i, j])
                edge_attr_list.append([
                    dist / _MAX_EDGE_DIST_NM,
                    math.sin(b_ij),
                    math.cos(b_ij),
                    math.sin(rel_hdg),
                    math.cos(rel_hdg),
                ])

        # Aircraft–airport (bidirectional)
        for i in range(n):
            dist   = get_distance_nm(lats[i], lons[i], self.airport_lat, self.airport_lon)
            b_ap   = math.radians(get_azimuth_to_point(lats[i], lons[i], self.airport_lat, self.airport_lon))
            b_from = math.radians(get_azimuth_to_point(self.airport_lat, self.airport_lon, lats[i], lons[i]))
            d_norm = dist / _TERMINAL_RADIUS_NM

            edge_index_list.append([i, airport_idx])
            edge_attr_list.append([d_norm, math.sin(b_ap),   math.cos(b_ap),   0.0, 0.0])

            edge_index_list.append([airport_idx, i])
            edge_attr_list.append([d_norm, math.sin(b_from), math.cos(b_from), 0.0, 0.0])

        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr  = torch.tensor(edge_attr_list,  dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, 5), dtype=torch.float32)

        return Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_type: str,
        airport_lat: float,
        airport_lon: float,
        device: str = "cpu",
        **kwargs,
    ) -> "GNNAgent":
        """Load a trained GNN model, auto-detecting hyperparameters from the checkpoint.

        Hyperparameters (``hidden_dim``, ``num_node_features``, etc.) are
        read directly from the state-dict tensor shapes so callers do not
        need to specify them.

        Args:
            checkpoint_path: Path to the ``.pt`` state-dict file.
            model_type:      ``"gat"`` or ``"gcn"``.
            airport_lat:     Airport reference latitude (degrees).
            airport_lon:     Airport reference longitude (degrees).
            device:          Torch device string.
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Unwrap training checkpoint wrapper if present
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        # Infer architecture from tensor shapes
        num_node_features = state_dict["node_encoder.weight"].shape[1]
        hidden_dim        = state_dict["node_encoder.weight"].shape[0]
        num_layers        = sum(1 for k in state_dict if k.startswith("batch_norms.") and k.endswith(".weight"))
        num_classes       = state_dict["classifier.3.weight"].shape[0]

        if model_type == "gat":
            # Edge dim from first conv layer
            num_edge_features = state_dict["convs.0.lin_edge.weight"].shape[1]
            # num_heads from att_src shape [1, heads, head_dim]
            num_heads = state_dict["convs.0.att_src"].shape[1]
            model: torch.nn.Module = GATHeadingPredictor(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                num_heads=num_heads,
                dropout=kwargs.get("dropout", 0.2),
            )
        elif model_type == "gcn":
            num_edge_features = state_dict.get(
                "edge_encoder.weight",
                torch.zeros(1, 5)
            ).shape[1]
            model = BaseGNN(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=kwargs.get("dropout", 0.2),
            )
        else:
            raise ValueError(f"Unknown model_type {model_type!r}; choose 'gat' or 'gcn'.")

        model.load_state_dict(state_dict)
        print(f"Loaded {model_type.upper()} from {checkpoint_path} "
              f"(nodes={num_node_features}, edges={num_edge_features}, "
              f"hidden={hidden_dim}, layers={num_layers}, classes={num_classes})")

        return cls(
            model=model,
            airport_lat=airport_lat,
            airport_lon=airport_lon,
            device=device,
            num_classes=num_classes,
        )


