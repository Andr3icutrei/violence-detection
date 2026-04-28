import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { InferenceActionResponseDto } from '../../core/api/models/inference-action-response-dto';
import { environment } from '../../../environments/environment.development';

@Injectable({
  providedIn: 'root',
})
export class InferenceActionsService {
  constructor(private httpClient: HttpClient) {}

  public getInferenceActionsForDataset(
    datasetId: number,
  ): Observable<InferenceActionResponseDto[]> {
    return this.httpClient.get<InferenceActionResponseDto[]>(
      `${environment.apiUrl}inference_actions/get_inference_actions_for_dataset/${datasetId}`,
      { withCredentials: true },
    );
  }

  public getInferenceActionsStats(): Observable<InferenceActionResponseDto[]> {
    return this.httpClient.get<InferenceActionResponseDto[]>(`${environment.apiUrl}inference_actions/get_inference_actions_stats`, { withCredentials: true });
  }

  public updateCreditsForAction(inferenceActionId: number, newCredits: number): Observable<void> {
    const params = new HttpParams()
      .set('credits', newCredits)
      .set('inferenceActionId', inferenceActionId);
    return this.httpClient.patch<void>(
      `${environment.apiUrl}inference_actions/${inferenceActionId}`,
      null,
      {
        withCredentials: true,
        params: params,
      },
    );
  }
}
