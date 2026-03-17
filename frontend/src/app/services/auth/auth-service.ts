import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { environment } from '../../../environments/environment.development';
import { Observable } from 'rxjs';
import { UserResponseDto } from '../../core/api/models/user-response-dto';

@Injectable({
  providedIn: 'root',
})
export class AuthService {
  constructor(private http: HttpClient) {}

  public login(email: string, password: string): Observable<UserResponseDto> {
    if (!email || !password) {
      throw new Error('Email and password are required for registration.');
    }
    const body = {
      email: email,
      password: password,
    };
    return this.http.post<UserResponseDto>(environment.apiUrl + 'auth/login', body, {withCredentials: true});
  }

  public logout(): Observable<void> {
    return this.http.post<void>(environment.apiUrl + 'auth/logout', null, { withCredentials: true });
  }

  public me(): Observable<UserResponseDto> {
    return this.http.get<UserResponseDto>(environment.apiUrl + 'auth/me',  { withCredentials: true });
  }
}
