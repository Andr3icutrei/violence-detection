import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { InferenceHistoryStatsResponseDto } from '../../core/api/models/inference-history-stats-response-dto';
import { environment } from '../../../environments/environment.development';

@Injectable({
  providedIn: 'root',
})
export class InferenceHistoryService {
  constructor(private httpClient: HttpClient) {}

  public getInferenceHistoryStats(year?: number, month?: number): Observable<InferenceHistoryStatsResponseDto> {
    let params = new HttpParams();
    if (year) {
      params = params.set('year', year);
    }
    if (month) {
      params = params.set('month', month);
    }
    return this.httpClient.get<InferenceHistoryStatsResponseDto>(environment.apiUrl + 'inference_history/get_inference_history_stats', { withCredentials: true, params: params });
  }
}
