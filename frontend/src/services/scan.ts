import api from "./api";

export interface SpeciesPrediction {
  name: string;
  confidence: number;
}

export interface DiseaseResult {
  available: boolean;
  name: string | null;
  confidence: number | null;
  is_healthy: boolean | null;
  severity: string | null;
}

export interface ScanResult {
  id: number;
  user_id: number;
  image_path: string;

  species: {
    name: string;
    confidence: number;
    top_3: SpeciesPrediction[];
  };

  disease: DiseaseResult;

  gradcam_image_path: string | null;
  growth_stage: string | null;
  recommendations: unknown;
  created_at: string | null;
}

export interface ScanResponse {
  message: string;
  scan: ScanResult;
}

export const scanPlant = async (
  image: File
): Promise<ScanResponse> => {
  const formData = new FormData();

  formData.append("image", image);

  const response = await api.post<ScanResponse>(
    "/scan",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );

  return response.data;
};

export interface ScanHistoryResponse {
  count: number;
  scans: ScanResult[];
}

export const getScanHistory = async (): Promise<ScanHistoryResponse> => {
  const response = await api.get<ScanHistoryResponse>(
    "/scan/history"
  );

  return response.data;
};