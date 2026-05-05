import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment.development';

@Injectable({
  providedIn: 'root',
})
export class CreditsService {
  constructor(private httpClient: HttpClient) {}

  public getCreditsCronjobUpdate(): Observable<number> {
    return this.httpClient.get<number>(`${environment.apiUrl}credits/get_credits_cronjob_update`, {
      withCredentials: true,
    });
  }

  public patchCreditsCronjobUpdate(newCredits: number): Observable<void> {
    const params = new HttpParams().set('new_credits', newCredits);
    return this.httpClient.patch<void>(
      `${environment.apiUrl}credits/patch_credits_cronjob_update`,
      null,
      { withCredentials: true, params: params },
    );
  }
}
