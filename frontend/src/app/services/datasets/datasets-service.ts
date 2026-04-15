import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { DatasetResponseDto } from '../../core/api/models/dataset-response-dto';
import { environment } from '../../../environments/environment.development';

@Injectable({
  providedIn: 'root',
})
export class DatasetsService {
  constructor(private httpClient: HttpClient) {}

  public getDatasets(): Observable<DatasetResponseDto[]> {
    return this.httpClient.get<DatasetResponseDto[]>(environment.apiUrl + 'datasets/get_datasets', {
      withCredentials: true,
    });
  }

  public createUnofficialDataset(formData: FormData): Observable<void> {
    return this.httpClient.post<void>(`${environment.apiUrl}datasets/create_unofficial_dataset`, formData, {
      withCredentials: true,
    });
  }
}
