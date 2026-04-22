import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { InferenceActionResponseDto } from '../../core/api/models/inference-action-response-dto';
import { environment } from '../../../environments/environment.development';

@Injectable({
  providedIn: 'root',
})
export class InferenceActionsService {
  constructor(private httpClient: HttpClient) {

  }

  public getInferenceActions(datasetId: number): Observable<InferenceActionResponseDto[]> {
    return this.httpClient.get<InferenceActionResponseDto[]>(
      `${environment.apiUrl}inference_actions/get_inference_actions_for_dataset/${datasetId}`,
      { withCredentials: true },
    );
  }
}
