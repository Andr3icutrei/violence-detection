import { ComponentFixture, TestBed } from '@angular/core/testing';

import { VerifyAccountServerError } from './verify-account-server-error';

describe('VerifyAccountServerError', () => {
  let component: VerifyAccountServerError;
  let fixture: ComponentFixture<VerifyAccountServerError>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [VerifyAccountServerError],
    }).compileComponents();

    fixture = TestBed.createComponent(VerifyAccountServerError);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
