import { Injectable } from '@angular/core';
import { HttpClient, HttpParams, HttpResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { VideoResponseDto } from '../../core/api/models/video-response-dto';
import { environment } from '../../../environments/environment.development';

@Injectable({
  providedIn: 'root',
})
export class VideosService {
  constructor(private httpClient: HttpClient) {}

  public getVideosPaged(
    asc: boolean,
    page: number,
    page_size: number,
    search_term?: string,
    dataset_id?: number,
  ): Observable<VideoResponseDto[]> {
    let params = new HttpParams().set('page', page).set('page_size', page_size).set('asc', asc);
    if (search_term) {
      params = params.set('search_term', search_term);
    }
    if (dataset_id && dataset_id !== 0) {
      params = params.set('dataset_id', dataset_id);
    }
    return this.httpClient.get<VideoResponseDto[]>(`${environment.apiUrl}videos/get_videos_paged`, {
      params: params,
      withCredentials: true,
    });
  }

  public existsVideo(uid: string): Observable<boolean> {
    return this.httpClient.get<boolean>(`${environment.apiUrl}videos/exists_video/${uid}`, {
      withCredentials: true,
    });
  }

  public inferenceVideo(videoId: number, selectedActionId: number): Observable<HttpResponse<Blob>> {
    const responseOptions = {
      withCredentials: true,
      observe: 'response' as const,
      responseType: 'blob' as const,
    };

    if (selectedActionId === 10) {
      return this.httpClient.post(
        `${environment.apiUrl}videos/classify_video_gradcam/${videoId}`,
        null,
        responseOptions,
      );
    } else if (selectedActionId === 20) {
      return this.httpClient.post(
        `${environment.apiUrl}videos/people_tracking/${videoId}`,
        null,
        responseOptions,
      );
    }

    return throwError(() =>
      new Error(`Unsupported inference action id received: ${selectedActionId}`),
    );
  }
}
