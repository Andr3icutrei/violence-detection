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
    return this.httpClient.get<InferenceActionResponseDto[]>(
      `${environment.apiUrl}inference_actions/get_inference_actions_stats`,
      { withCredentials: true },
    );
  }

  public updateCreditsForAction(actions: { id: number; newCredits: number }[]): Observable<void> {
    if (actions === null || actions.length === 0) {
      throw new Error('No actions provided for updating credits.');
    }
    const body = {
      actions: actions.map((action) => ({
        id: action.id,
        new_credits: action.newCredits,
      })),
    };
    return this.httpClient.patch<void>(
      `${environment.apiUrl}inference_actions/update_credits_inference_actions`,
      body,
      {
        withCredentials: true,
      },
    );
  }
}
