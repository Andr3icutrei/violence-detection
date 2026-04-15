import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { DatasetResponseDto } from '../../core/api/models/dataset-response-dto';
import { environment } from '../../../environments/environment.development';

@Injectable({
  providedIn: 'root',
})
export class DatasetsService {
  constructor(private httpClient: HttpClient) {}

  public getAcceptedDatasets(): Observable<DatasetResponseDto[]> {
    return this.httpClient.get<DatasetResponseDto[]>(environment.apiUrl + 'datasets/get_accepted_datasets', {
      withCredentials: true,
    });
  }

  public createUnofficialDataset(formData: FormData): Observable<void> {
    return this.httpClient.post<void>(`${environment.apiUrl}datasets/create_unofficial_dataset`, formData, {
      withCredentials: true,
    });
  }

  public getPendingDatasets(searchTerm: string, page: number, pageSize: number): Observable<DatasetResponseDto[]> {
    const params: HttpParams = new HttpParams()
      .set('page', page)
      .set('page_size', pageSize)
      .set('search_term', searchTerm);

    return this.httpClient.get<DatasetResponseDto[]>(environment.apiUrl + 'datasets/get_pending_datasets', {
      withCredentials: true,
    });
  }
}
