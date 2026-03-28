import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { DatasetResponseDto } from '../../core/api/models/dataset-response-dto';
import { environment } from '../../../environments/environment.development';

@Injectable({
  providedIn: 'root',
})
export class DatasetsService {
  constructor(private httpClient: HttpClient) {

  }

  public getDatasets(): Observable<DatasetResponseDto[]> {
    return this.httpClient.get<DatasetResponseDto[]>(environment.apiUrl + 'datasets/get_datasets', {
      withCredentials: true,
    });
  }
}
